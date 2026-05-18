# import sys
# from pathlib import Path
# import numpy as np
# import trimesh
# from trimesh.transformations import translation_matrix, rotation_matrix, concatenate_matrices
# from scipy.spatial import ConvexHull
# from shapely.geometry import Polygon
#
#
# # ─────────────────────────────────────────────────────────────
# # 1. Утилиты
# # ─────────────────────────────────────────────────────────────
# def get_file_path(filename):
#     """Получает путь к файлу, совместимый с PyInstaller"""
#     base_path = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
#     return base_path / filename
#
#
# def get_contour_bounds(contour):
#     """Безопасное получение ширины, высоты и центра контура"""
#     pts = np.array(contour).reshape(-1, 2)
#     min_xy = pts.min(axis=0)
#     max_xy = pts.max(axis=0)
#     width = max_xy[0] - min_xy[0]
#     height = max_xy[1] - min_xy[1]
#     center = (min_xy + max_xy) / 2
#     return width, height, center
#
#
# def transform_mesh(mesh, translation, rotation=None, scale=None):
#     """Универсальное применение трансформаций (Scale -> Rotate -> Translate)"""
#     m = mesh.copy()
#     if scale is not None:
#         m.apply_scale(scale)
#     if rotation is not None:
#         m.apply_transform(rotation)
#     m.apply_translation(translation)
#     return m
#
#
# def gen_floor(polygon_points, thickness=0.3, color=(150, 150, 250, 200)):
#     """Создание пола из 2D полигона"""
#     polygon = Polygon(polygon_points)
#     floor = trimesh.creation.extrude_polygon(polygon, height=thickness)
#     # Поднимаем пол на его толщину/2, чтобы верхняя грань была на Z=0
#     floor.apply_translation([0, 0, -thickness / 2])
#     matrix_z_inversion = np.array([
#         [1, 0, 0, 0],
#         [0, -1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])
#     floor.apply_transform(matrix_z_inversion)
#     floor.visual.face_colors = color
#     return floor
#
#
# # ─────────────────────────────────────────────────────────────
# # 2. Единая функция размещения проёмов с CSG
# # ─────────────────────────────────────────────────────────────
# def place_openings(openings, wall_meshes, scale, height, base_mesh,
#                    pre_rotation=None, wall_thickness=0.2, config=None):
#     """
#     Вырезает проёмы в стенах и размещает объекты.
#     :param openings: список контуров
#     :param wall_meshes: список мешей стен (будут изменены на месте)
#     :param scale: масштаб пиксели -> метры
#     :param height: высота помещения
#     :param base_mesh: базовый меш объекта (дверь/окно)
#     :param pre_rotation: начальная ориентация базового меша
#     :param wall_thickness: толщина стены для выреза
#     """
#     if config is None:
#         config = {'offset': [0.0, 0.0, 0.0], 'bottom_z': 0.0, 'manual_pos': None}
#
#     objects = []
#
#     for i, contour in enumerate(openings):
#         try:
#             # 1. Определяем позицию (ручную или из контура)
#             if config.get('manual_pos') is not None:
#                 pos_3d = np.array(config['manual_pos'], dtype=float)
#                 w_3d, h_3d = config.get('manual_size', [1.0, 1.0])
#             else:
#                 w_px, h_px, center_px = get_contour_bounds(contour)
#                 if w_px <= 0 or h_px <= 0:
#                     continue
#                 pos_3d = np.array([center_px[0] * scale, -center_px[1] * scale, height / 2])
#                 w_3d, h_3d = w_px * scale, h_px * scale
#
#             # 2. Применяем смещение
#             offset = np.array(config.get('offset', [0, 0, 0]), dtype=float)
#             pos_3d += offset
#
#             # 5. Готовим объект: Масштаб -> Поворот -> Точная позиция
#             obj = base_mesh.copy()
#
#             # Масштаб под проём
#             obj_ext = obj.extents
#             if obj_ext[0] > 0 and obj_ext[2] > 0:
#                 obj.apply_scale([w_3d / obj_ext[0], 1.0, h_3d / obj_ext[2]])
#
#             # Поворот (если нужен)
#             if pre_rotation is not None:
#                 obj.apply_transform(pre_rotation)
#
#             # Финальная позиция: учитываем bottom_z (высота нижней грани от пола)
#             bottom_z = config.get('bottom_z', 0.0)
#             final_pos = pos_3d.copy()
#             final_pos[2] = bottom_z + obj.extents[2] / 2  # Поднимаем так, чтобы низ был на bottom_z
#
#             # 3. Создаём вырез (синхронизирован с объектом)
#             cutout = trimesh.primitives.Box(extents=[w_3d, wall_thickness + 0.02, h_3d])
#             # Позиционируем вырез так, чтобы его центр совпадал с центром объекта по Z
#             cutout.apply_translation(final_pos)
#
#             # 4. Вычитаем из стен
#             for j, wall in enumerate(wall_meshes):
#                 c_min, c_max = cutout.bounds
#                 w_min, w_max = wall.bounds
#                 if (c_min[0] < w_max[0] and c_max[0] > w_min[0] and
#                         c_min[1] < w_max[1] and c_max[1] > w_min[1]):
#                     try:
#                         wall_meshes[j] = trimesh.boolean.difference(
#                             [wall, cutout], engine="manifold"
#                         )
#                     except Exception as e:
#                         print(f"⚠️ CSG failed for wall {j}: {e}")
#
#             # Переносим объект в итоговую позицию
#             obj.apply_translation(final_pos - obj.centroid)
#
#             objects.append(obj)
#             print(f"✅ Проём #{i + 1} размещён на Z={bottom_z} м, позиция: {final_pos}")
#
#         except Exception as e:
#             print(f"❌ Ошибка обработки проёма #{i + 1}: {e}")
#             continue
#
#     return objects
#
#
# # ─────────────────────────────────────────────────────────────
# # 3. Сборка сцены
# # ─────────────────────────────────────────────────────────────
# def build_3d_model(wall_contours, original_size, scale=0.1, height=3.0,
#                    doors=None, windows=None):
#     """
#     Основная функция сборки. Поддерживает CSG и легко расширяется.
#     """
#     scene_objects = []
#     all_points = []
#
#     # 1. Строим стены
#     wall_meshes = []
#     for contour in wall_contours:
#         pts = np.array(contour).reshape(-1, 2)
#         if len(pts) < 3:
#             continue
#
#         scaled = pts * float(scale)
#         if not np.allclose(scaled[0], scaled[-1]):
#             scaled = np.vstack([scaled, scaled[0]])
#
#         matrix_z_inversion = np.array([
#                         [1, 0, 0, 0],
#                         [0, -1, 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]
#                     ])
#
#         all_points.append(scaled)
#
#         try:
#             wall = trimesh.creation.extrude_polygon(Polygon(scaled), height=height)
#             # Поднимаем, чтобы пол был на Z=0
#             wall.apply_translation([0, 0, 0])
#             wall.apply_transform(matrix_z_inversion)
#             wall_meshes.append(wall)
#         except Exception as e:
#             print(f"⚠️ Ошибка стены: {e}")
#
#     # 2. Вырезаем проёмы и размещаем объекты
#     if doors and mesh_door:
#         rot_y = rotation_matrix(np.pi / 2, [1, 0, 0])
#         scene_objects.extend(place_openings(doors, wall_meshes, scale, height, mesh_door,pre_rotation=rot_y, config=OPENING_PRESETS['door']))
#     if windows and mesh_window:
#         # Окна обычно повёрнуты на 90° относительно дверей, корректируем
#         rot_y = rotation_matrix(np.pi / 2, [1, 0, 0])
#         # scene_objects.extend(place_openings(windows, wall_meshes, scale, height, mesh_window, pre_rotation=rot_y))
#         scene_objects.extend(place_openings(windows, wall_meshes, scale, height, mesh_window,
#                        pre_rotation=rot_y, config=OPENING_PRESETS['window_standard']))
#
#     # 3. Добавляем модифицированные стены и пол
#     scene_objects.extend(wall_meshes)
#
#     if all_points:
#         hull_points = np.vstack(all_points)
#         hull = ConvexHull(hull_points)
#         floor_poly = hull_points[hull.vertices]
#         scene_objects.append(gen_floor(floor_poly))
#
#     if not scene_objects:
#         raise ValueError("Не удалось создать ни одного 3D-объекта")
#
#     return trimesh.Scene(scene_objects)
#
#
# # ─────────────────────────────────────────────────────────────
# # 4. Инициализация (пример использования)
# # ─────────────────────────────────────────────────────────────
# obj_path_door = get_file_path("3d_obj_test/Door_Component.obj")
# obj_path_win = get_file_path("3d_obj_test/window1.obj")
# mesh_door = trimesh.load(obj_path_door) if obj_path_door.exists() else None
# mesh_window = trimesh.load(obj_path_win) if obj_path_win.exists() else None
#
# OPENING_PRESETS = {
#     'door': {'bottom_z': 0.0, 'offset': [0, 0, 0]},
#     'window_standard': {'bottom_z': 0.9, 'offset': [0, 0, 0]},
#     'window_high': {'bottom_z': 1.8, 'offset': [0, 0, 0]},
#     'ventilation': {'bottom_z': 2.2, 'offset': [0, 0, 0]},
# }


# Пример вызова:
# scene = build_3d_model(wall_contours, img_size, scale=0.05, height=2.7,
#                        doors=door_contours, windows=win_contours)
# scene.show()



# ________________________________________________________________________________________________________________________
import sys
from pathlib import Path
import numpy as np
import trimesh

import img2wall as i2w
from scipy.spatial import ConvexHull
from trimesh.transformations import rotation_matrix, concatenate_matrices
from shapely.geometry import Polygon

import logging

from check_ddor import extract_wall_segments_with_ids, find_true_opening_center

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
# logger = logging.getLogger(__name__)

def get_file_path(filename: str) -> Path:
    base_path = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
    return base_path / filename

# obj_path_door = get_file_path("3d_obj_test/Door_Component.obj")
# obj_path_win = get_file_path("3d_obj_test/window1.obj")
#
# mesh_door = trimesh.load(obj_path_door)
# mesh_window = trimesh.load(obj_path_win)


def clear_cache():
    """Очистка кэшей генератора 3D"""
    import gc

    # Очистка внутреннего кэша trimesh (если доступен)
    if hasattr(trimesh, 'cache') and hasattr(trimesh.cache, 'clear'):
        trimesh.cache.clear()

    # Также можно очистить кэш загрузчиков
    if hasattr(trimesh.exchange, 'load') and hasattr(trimesh.exchange.load, '_mesh_loaders'):
        # Не трогаем напрямую, но можно пересоздать сцену
        pass

    # Принудительный сборщик мусора для крупных объектов
    gc.collect()


def _load_mesh_safe(path: Path, default_scale: float = 8.0):
    """Безопасная загрузка меша с валидацией."""
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh.is_empty:
            # logger.error(f"❌ Пустой меш: {path}")
            print(f"❌ Пустой меш: {path}")
            return None
        mesh.apply_scale(default_scale)
        mesh.fix_normals()
        # logger.info(f"✅ Загружен меш: {path.name}, вершин: {len(mesh.vertices)}")
        print(f"✅ Загружен меш: {path.name}, вершин: {len(mesh.vertices)}")
        return mesh
    except Exception as e:
        # logger.error(f"❌ Ошибка загрузки {path}: {e}")
        print(f"❌ Ошибка загрузки {path}: {e}")
        return None

# Door_MESH = _load_mesh_safe(get_file_path())
# Window_MESH = _load_mesh_safe(get_file_path())
# _FRI_MESH = _load_mesh_safe(get_file_path())
# _KitSink_MESH = _load_mesh_safe(get_file_path("3d_obj_test/Separate_assets_obj/kitchen_sink_001.obj"))
# _WasMach_MESH = _load_mesh_safe(get_file_path("3d_obj_test/Separate_assets_obj/washing_machine_001.obj"))

obj_mesh = {'Door': "3d_obj_test/Door_Component.obj",
            'GasPlate': "3d_obj_test/Separate_assets_obj/kitchen_table_001.obj",
            'Wardor': "",
            'Wall': "",
            # 'Window': "3d_obj_test/3d models/windows_1sector.obj",
            'Window': "3d_obj_test/window1.obj",
            'bathtube': "3d_obj_test/OBJ/bath.obj",
            'box': "3d_obj_test/Separate_assets_obj/box_001.obj",
            'cold_box': "3d_obj_test/Separate_assets_obj/fridge_001.obj",
            'door-s': "",
            'door_l': "",
            'door_balcon': "",
            'door_bath': "3d_obj_test/Separate_assets_obj/door_001.obj",
            'h-wall': "",
            'door_vhod_l': "",
            'sink': "3d_obj_test/OBJ/sink.obj",
            'sink_kitchen': "3d_obj_test/Separate_assets_obj/kitchen_sink_001.obj",
            'balcon_wall': "",
            'toulet': "3d_obj_test/OBJ/toilet.obj",
            'wash_machine': "3d_obj_test/Separate_assets_obj/washing_machine_001.obj",
            'win_in_wall': ""
}

def gen_floor(objects):

    polygon = Polygon(objects)

    # 4. Создаем 3D меш пола
    floor = trimesh.creation.extrude_polygon(polygon, height=0.3)

    matrix_z_inversion = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])
    floor.apply_transform(matrix_z_inversion)

    # Настраиваем внешний вид
    floor.visual.face_colors = [150, 150, 250, 200]  # Полупрозрачный синий

    # Создаем сцену
    return floor



def build_obj(obj_contours, wall_contours, scale):
    scene_objects = []
    # logging.info(f'Всё - {obj_contours}')

    for t, class_o in enumerate(obj_contours):
        if (class_o == "Door" or class_o == "Window"):
            continue
        if obj_mesh[class_o]:
            logging.info(f'Tekuschiq - {class_o}')
            mesh_obj = _load_mesh_safe(get_file_path(obj_mesh[class_o]))

            for i, contour in enumerate(obj_contours[class_o]):
                # logger.info(")()()()(try copy mesh)()()()()(")
                # print(")()()()(try copy mesh)()()()()(")

                mesh = mesh_obj.copy()
                # print(")()()()( copy pass mesh)()()()()(")

                # wight_d = contour[2][0] - contour[0][0]
                # hight_d = contour[1][1] - contour[0][1]

                # ppoint = contour[0]
                # p_ch = False
                # for i1, contur in enumerate(wall_contours):
                #     next_wall = wall_contours[(i1+1) % len(wall_contours)]
                    # box_door = check_door(contur, next_wall)
                    # for p, ponit in enumerate(box_door):
                    #     for op in range(4):
                    #         if abs(ponit[0] - contour[op][0]) < 8:
                    #             # print(                            f"Точка {p} =[{ponit[0], ponit[1]}] b Точка{i1} =[{contour[op][0], contour[op][1]}] - близки X")
                    #             # contour[op][0] = ponit[0]
                    #             ppoint = ponit
                    #             p_ch = True
                    #         if abs(ponit[1] - contour[op][1]) < 8:
                    #             # print(                            f"Точка {p} =[{ponit[0], ponit[1]}] b Точка{i1} =[{contour[op][0], contour[op][1]}] - близки Y")
                    #             # contour[op][1] = ponit[1]
                    #             ppoint = ponit
                    #             p_ch = True

                # print(f")()()()( kontur)()() = ()()({contour}")

                # contour = i2w.average_close_points(contour, 300)
                # contour = np.column_stack((contour, np.zeros(len(contour))))

                # Преобразуем контур в массив точек
                contour_points = np.array(contour[0])
                # print(contour_points)
                # print(ppoint)

                # Масштабируем координаты
                scaled = contour_points * float(scale)
                # ppoint_s = ppoint * float(scale)
                # print(scaled)
                # print(ppoint_s)
                # if wight_d > hight_d and p_ch:
                #     scaled = scaled[0], ppoint_s[1], scaled[2]
                # if wight_d < hight_d and p_ch:
                #     scaled = ppoint_s[0], scaled[1], scaled[2]
                # print(scaled)
                # print(scaled[0])
                # print(translation_matrix(scaled))

                matrix_z_inversion = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ])

                try:
                    matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), 0  # Устанавливаем смещение
                    # matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1] + (hight_d * float(scale))/2), 0  # Устанавливаем смещение
                    # Создаём 2D полигон
                    # mesh.apply_scale(8)
                    # hight_win = mesh.extents.tolist()
                    # if wight_d > hight_d:
                    #     scale_win = wight_d*float(scale) / hight_win[0]
                    #     # print("wight_d = ", wight_d*float(scale))
                    # else:
                    #     scale_win = hight_d*float(scale) / hight_win[0]
                    #     # print("hight_d = ", hight_d*float(scale))
                    # # print(hight_win)
                    # # print(scale_win)
                    # mesh.apply_scale([scale_win, 1, 1])
                    rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
                    transform = concatenate_matrices(matrix_z_inversion, rot)
                    mesh.apply_transform(transform)
                    # # print(transform)
                    # if wight_d < hight_d:
                    #     # 1. Находим его центр (bounding box центроид)
                    #     center = mesh.bounding_box.centroid
                    #
                    #     # 2. Переносим в начало координат
                    #     mesh.apply_translation(-center)
                    #
                    #     # 3. Поворачиваем (например, на 45° вокруг оси Z)
                    #     rotation = rotation_matrix(np.pi / 2, [0, 0, 1])  # Ось Z
                    #     mesh.apply_transform(rotation)
                    #
                    #     # 4. Возвращаем на исходную позицию
                    #     mesh.apply_translation(center)
                    #
                    # hight_door = mesh.extents.tolist()
                    # # print(hight_door)
                    # h_d_p = float(height) - hight_door[2]
                    # box = trimesh.primitives.Box(extents=[hight_door[0], hight_door[1], h_d_p])
                    # matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), float(height) - h_d_p/2 # Устанавливаем смещение
                    #
                    # box.apply_transform(matrix_z_inversion)
                    #
                    # # Добавляем в сцену
                    # print(")()()()(try add mesh)()()()()(")
                    scene_objects.append(mesh)
                    # scene_objects.append(box)

                    print(f"Контур {class_o} #{i + 1} успешно преобразован в 3D-объект")

                except Exception as e:
                    print(f"Ошибка обработки контура {class_o} #{i + 1}: {str(e)}")
                    continue
    return scene_objects

# def check_door(wall1, wall2):
#     box = []
#     for i1, point1 in enumerate(wall1):
#         x1,y1 = point1.ravel()
#         for i2, point2 in enumerate(wall2):
#             x2, y2 = point2.ravel()
#             wall_length = np.linalg.norm(point1 - point2)
#             if ((x1 == x2) or (x1 == x2 -1) or (x1 == x2+1)) and 150.0<abs(wall_length)<220.0:
#                 # print(f"Точка {i1} =[{x1, y1}] b Точка{i2} =[{x2, y2}]: длина стены = {wall_length:.2f} м")
#                 box.append(point1)
#                 box.append(point2)
#                 # print(point2)
#                 # print(point1)
#             else:
#                 if ((y1 == y2) or (y1 == y2 -1) or (y1 == y2+1)) and 150.0<abs(wall_length)<220.0:
#                     # print(f"Точка {i1} =[{x1, y1}] b Точка{i2} =[{x2, y2 }]: длина стены = {wall_length:.2f} м")
#                     box.append(point1)
#                     box.append(point2)
#                     # print(point2)
#                     # print(point1)
#     # print("door")
#     # print(box)
#     return box

# def build_object(obj_contours, wall_contours, scale, height = 2.7):

def build_door(door_contours, wall_contours, scale, height = 2.7):
    scene_objects = []
    segments = extract_wall_segments_with_ids(wall_contours)
    mesh_door = _load_mesh_safe(get_file_path(obj_mesh['Door']), 1)
    for i, contour in enumerate(door_contours):
        mesh = mesh_door.copy()
        wight_d = contour[2][0] - contour[0][0]
        hight_d = contour[1][1] - contour[0][1]

        # ppoint = contour[0]
        # p_ch = False
        # for i1, contur in enumerate(wall_contours):
        #     next_wall = wall_contours[(i1+1) % len(wall_contours)]
            # box_door = check_door(contur, next_wall)
            # for p, ponit in enumerate(box_door):
            #     for op in range(4):
            #         if abs(ponit[0] - contour[op][0]) < 8:
            #             # print(                            f"Точка {p} =[{ponit[0], ponit[1]}] b Точка{i1} =[{contour[op][0], contour[op][1]}] - близки X")
            #             # contour[op][0] = ponit[0]
            #             ppoint = ponit
            #             p_ch = True
            #         if abs(ponit[1] - contour[op][1]) < 8:
            #             # print(                            f"Точка {p} =[{ponit[0], ponit[1]}] b Точка{i1} =[{contour[op][0], contour[op][1]}] - близки Y")
            #             # contour[op][1] = ponit[1]
            #             ppoint = ponit
            #             p_ch = True

        contour = i2w.average_close_points(contour, 300)
        contour = np.column_stack((contour, np.zeros(len(contour))))

        # Преобразуем контур в массив точек
        contour_points = np.array(contour[0])
        # print(contour_points)
        # print(ppoint)

        # Масштабируем координаты
        scaled = contour_points * float(scale)
        # ppoint_s = ppoint * float(scale)
        # print(scaled)
        # print(ppoint_s)
        # if wight_d > hight_d and p_ch:
        #     scaled = scaled[0], ppoint_s[1], scaled[2]
        # if wight_d < hight_d and p_ch:
        #     scaled = ppoint_s[0], scaled[1], scaled[2]
        # print(scaled)
        # print(scaled[0])
        # print(translation_matrix(scaled))

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        try:
            matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), 0  # Устанавливаем смещение
            # matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1] + (hight_d * float(scale))/2), 0  # Устанавливаем смещение
            # Создаём 2D полигон
            mesh.apply_scale(8)
            hight_win = mesh.extents.tolist()
            if wight_d > hight_d:
                scale_win = wight_d*float(scale) / hight_win[0]
                # print("wight_d = ", wight_d*float(scale))
            else:
                scale_win = hight_d*float(scale) / hight_win[0]
                # print("hight_d = ", hight_d*float(scale))
            # print(hight_win)
            # print(scale_win)
            mesh.apply_scale([scale_win, 1, 1])
            rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
            transform = concatenate_matrices(matrix_z_inversion, rot)
            mesh.apply_transform(transform)
            # print(transform)
            if wight_d < hight_d:
                # 1. Находим его центр (bounding box центроид)
                center = mesh.bounding_box.centroid

                # 2. Переносим в начало координат
                mesh.apply_translation(-center)

                # 3. Поворачиваем (например, на 45° вокруг оси Z)
                rotation = rotation_matrix(np.pi / 2, [0, 0, 1])  # Ось Z
                mesh.apply_transform(rotation)

                # 4. Возвращаем на исходную позицию
                # print(f"центр пов{center}")
                mesh.apply_translation(center)

            # Находим центр двери (в пикселях)
            door_center_px = np.array([
                (contour[:, 0].min() + contour[:, 0].max()) / 2,
                (contour[:, 1].min() + contour[:, 1].max()) / 2
            ])

            # === 2. Привязка к проёму в стене ===
            result = find_true_opening_center(door_center_px, segments, search_radius=300)

            if result:
                # Используем центр проёма, а не детектированный центр двери
                final_center_px = np.array(result['center_px'])
                door_angle_rad = np.radians(result['angle_deg'])
                opening_width_px = result['opening_width_px']
                # print(f"Дверь #{i + 1}: центр проёма {final_center_px}, Ширина проёма: {opening_width_px}, угол {door_angle_rad}°")
            else:
                # Если проём не найден, используем центр детектированной двери
                final_center_px = door_center_px
                door_angle_rad = 0
                opening_width_px = wight_d
                # print(f"Дверь #{i + 1}: проём не найден, используем детектированный центр")
                # print(f"Дверь #{i + 1}: центр проёма {final_center_px}, Ширина проёма: {opening_width_px}, угол {door_angle_rad}°")

            # === 3. Масштабирование в мировые координаты ===
            final_center_world = final_center_px * float(scale)
            door_width_world = wight_d * float(scale)
            door_height_world = hight_d * float(scale)
            # print(f"Дверь #{i + 1}: центр проёма {final_center_world}, Ширина проёма: {door_width_world}, hight {door_height_world}°")


            hight_door = mesh.extents.tolist()
            # print(hight_door)
            h_d_p = float(height) - hight_door[2]
            box = trimesh.primitives.Box(extents=[hight_door[0], hight_door[1], h_d_p])
            matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), float(height) - h_d_p/2 # Устанавливаем смещение

            box.apply_transform(matrix_z_inversion)

            # Добавляем в сцену
            scene_objects.append(mesh)
            scene_objects.append(box)

            print(f"Контур DOOR #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура DOOR #{i + 1}: {str(e)}")
            continue
    return scene_objects

def build_window(window_position, wall_contours, scale, height = 2.7):
    scene_objects = []
    mesh_window = _load_mesh_safe(get_file_path(obj_mesh['Window']), 1)
    for i, contour in enumerate(window_position):
        win_mesh = mesh_window.copy()
        wight_d = contour[2][0] - contour[0][0]
        hight_d = contour[1][1] - contour[0][1]

        contour = i2w.average_close_points(contour, 300)
        contour = np.column_stack((contour, np.zeros(len(contour))))

        # print(contour)
        # Преобразуем контур в массив точек
        contour_points = np.array(contour[0])

        # Масштабируем координаты
        scaled = contour_points * float(scale)
        # print(scaled)
        # print(scaled[0])
        # print(translation_matrix(scaled))

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        try:
            matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), float(height)/2  # Устанавливаем смещение
            # matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1] + (hight_d * float(scale))/2), 0  # Устанавливаем смещение
            # Создаём 2D полигон
            win_mesh.apply_scale(0.1)
            hight_win = win_mesh.extents.tolist()
            if wight_d > hight_d:
                scale_win = wight_d*float(scale) / hight_win[2]
                # print("wight_d = ", wight_d)
            else:
                scale_win = hight_d*float(scale) / hight_win[2]
                # print("hight_d = ", hight_d)
            # print(hight_win)
            # print(scale_win)
            win_mesh.apply_scale([1, 1, scale_win])
            rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
            rot1 = rotation_matrix(np.pi / 2, [0, 1, 0])
            transform = concatenate_matrices(matrix_z_inversion, rot)
            transform = concatenate_matrices(transform, rot1)
            win_mesh.apply_transform(transform)


            # print(transform)
            if wight_d < hight_d:
                # 1. Находим его центр (bounding box центроид)
                center = win_mesh.bounding_box.centroid

                # 2. Переносим в начало координат
                win_mesh.apply_translation(-center)

                # 3. Поворачиваем (например, на 45° вокруг оси Z)
                rotation = rotation_matrix(np.pi / 2, [0, 0, 1])  # Ось Z
                win_mesh.apply_transform(rotation)

                # 4. Возвращаем на исходную позицию
                win_mesh.apply_translation(center)

            # Добавляем в сцену
            hight_win = win_mesh.extents.tolist()
            hight_win[2] = round(hight_win[2], 3)

            # print(hight_win)
            b_w_p = float(height)/2
            b_w_p = round(b_w_p, 3)
            h_w_p = float(height) - hight_win[2] - b_w_p
            h_w_p = round(h_w_p, 3)

            # print(f"высота = {height}, высота снизу = {b_w_p}, высота сверху = {h_w_p}, высота окна = {hight_win[2]}, высота сверху1 = {float(height) - h_w_p/2}")
            transformation = np.eye(4)
            box11 = trimesh.primitives.Box(extents=[hight_win[0], hight_win[1], h_w_p])
            h_w_p = float(height) - h_w_p/2
            transformation[:3, 3] = scaled[0], -(scaled[1]), h_w_p # Устанавливаем смещение
            box11.apply_transform(transformation)
            # print(box11.bounds)

            transformation = np.eye(4)
            box2 = trimesh.primitives.Box(extents=[hight_win[0], hight_win[1], b_w_p])
            transformation[:3, 3] = scaled[0], -(scaled[1]), b_w_p/2 # Устанавливаем смещение float(height) - h_d_p/2
            box2.apply_transform(transformation)

            scene_objects.append(win_mesh)
            scene_objects.append(box11)
            scene_objects.append(box2)

            print(f"Контур Window #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура Window #{i + 1}: {str(e)}")
            continue

    return scene_objects



# import numpy as np
# import trimesh
# from trimesh.transformations import rotation_matrix, translation_matrix, concatenate_matrices
#

#
# def build_opening(contours, base_mesh, scale: float, height: float = 2.7, is_door: bool = True, min_h: float = 0.9):
#     # float = 0.9
#     scene_objects = []
#     for i, contour in enumerate(contours):
#         try:
#             # 1. Подготовка контура
#             # pts = np.array(i2w.average_close_points(contour, 300))
#             width = abs(contour[2, 0] - contour[0, 0])
#             height_obj = abs(contour[1, 1] - contour[0, 1])
#
#             # 2. Базовая позиция (центр проёма)
#             center_2d = contour.mean(axis=0)
#             pos = [center_2d[0] * scale, -center_2d[1] * scale, 0.0]
#
#             # 3. Композиция трансформаций
#             T = translation_matrix(pos)
#             rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
#             T = concatenate_matrices(T, rot)
#
#             if width < height_obj:
#                 rot_z = rotation_matrix(np.pi / 2, [0, 0, 1])
#                 T = concatenate_matrices(T, rot_z)
#
#             # 4. Масштабирование
#             mesh = base_mesh.copy()
#             extents = mesh.extents  # вычисляется 1 раз
#             scale_factor = (width * scale) / extents[0] if is_door else (height_obj * scale) / extents[2]
#             S = np.diag([scale_factor, 1.0, 1.0] if is_door else [1.0, 1.0, scale_factor])
#             T[:3, :3] = T[:3, :3] @ S
#
#             mesh.apply_transform(T)
#
#             # 5. Заполнители (стена над/под проёмом)
#             if is_door:
#                 h_clear = height - mesh.extents[2]
#                 filler = trimesh.primitives.Box(extents=[mesh.extents[0], mesh.extents[1], h_clear])
#                 filler.apply_transform(translation_matrix([pos[0], pos[1], height - h_clear / 2]))
#                 scene_objects.extend([mesh, filler])
#             else:
#                 logging.debug(f"Проем #{mesh.extents}")
#                 h_clear = height - (mesh.extents[2] + min_h)
#                 logging.debug(f"h_clear #{h_clear}")
#                 filler1 = trimesh.primitives.Box(extents=[mesh.extents[0], mesh.extents[1], h_clear])
#                 filler2 = trimesh.primitives.Box(extents=[mesh.extents[0], mesh.extents[1], h_clear])
#                 filler1.apply_transform(translation_matrix([pos[0], pos[1], height - h_clear / 2]))
#                 filler2.apply_transform(translation_matrix([pos[0], pos[1], height - h_clear / 2]))
#                 scene_objects.extend([mesh, filler1, filler2])
#
#             logging.info(f"Проем #{i + 1} построен успешно")
#         except Exception as e:
#             logging.error(f"Ошибка в проеме #{i + 1}: {e}")
#             continue
#     return scene_objects

def build_3d_model(wall_contours, scale=0.1, height=3.0):
    """
    Строит 3D-модель помещения экструдированием контуров стен
    :param wall_contours: Контуры стен из обработки изображения
    :param original_size: Исходные размеры изображения (height, width)
    :param scale: Масштаб преобразования (пиксели в метры)
    :param height: Высота помещения в метрах
    :return: Сцена с 3D-моделью (trimesh.Scene)
    """

    scene_objects = []
    all_cont = []



    for i, contour in enumerate(wall_contours):
        # Преобразуем контур в массив точек
        contour_points = np.array(contour).reshape(-1, 2)

        # Упрощаем контур
        contour_points = contour_points.reshape(-1, 1, 2).astype(np.int32).squeeze()

        # Пропускаем контуры с малым количеством точек
        if len(contour_points) < 3:
            print(f"Контур #{i + 1} пропущен (мало точек: {len(contour_points)})")
            continue



        # Масштабируем координаты
        scaled = contour_points * float(scale)
        # print(scaled)
        all_cont.append(scaled)


        # Замыкаем контур при необходимости
        if not np.allclose(scaled[0], scaled[-1]):
            scaled = np.vstack([scaled, scaled[0]])

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Apply the transformation to the mesh vertices

        try:
            # Создаём 2D полигон
            polygon = trimesh.path.polygons.Polygon(scaled)
            # print("стена = ",scaled)
            # print("dscjnf стена = ", height)

            # Экструдируем в 3D
            mesh = trimesh.creation.extrude_polygon(polygon, height=height)
            mesh.apply_transform(matrix_z_inversion)

            # Добавляем в сцену
            scene_objects.append(mesh)

            print(f"Контур #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура #{i + 1}: {str(e)}")
            continue

    if not scene_objects:
        raise ValueError("Не удалось создать ни одного 3D-объекта!")

    # print(all_cont)
    all_cont = np.vstack(all_cont)

    # Находим выпуклую оболочку
    hull = ConvexHull(all_cont)

    # Получаем точки выпуклой оболочки (уже упорядочены)
    kr_p = all_cont[hull.vertices]

    # Замыкаем контур (добавляем первую точку в конец)
    kr_p = np.vstack([kr_p, kr_p[0]])

    scene_objects.append(gen_floor(kr_p))

    return trimesh.Scene(scene_objects)