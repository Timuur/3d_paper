import sys
from pathlib import Path
import numpy as np
import trimesh
import json

import time
from functools import wraps

def timing(func):
    """Декоратор для замера времени выполнения функции"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"⏱ {func.__name__}: {(t1-t0)*1000:.1f} ms" if (t1-t0) < 1 else f"⏱ {func.__name__}: {t1-t0:.2f} sec")
        return result
    return wrapper

import img2wall as i2w
from scipy.spatial import ConvexHull
from trimesh.transformations import rotation_matrix, concatenate_matrices
from shapely.geometry import Polygon

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from check_ddor import extract_wall_segments_with_ids, find_true_opening_center



from trimesh.visual.material import PBRMaterial, SimpleMaterial
import base64
from pathlib import Path

def _create_pbr_material(color=None, texture_path=None, metallic=0.1, roughness=0.8):
    """Создание PBR-материала для экспорта в GLTF."""
    if texture_path and Path(texture_path).exists():
        from PIL import Image
        image = Image.open(texture_path)
        return PBRMaterial(
            baseColorTexture=image,
            metallicFactor=metallic,
            roughnessFactor=roughness,
            doubleSided=True
        )
    else:
        return SimpleMaterial(
            diffuse=color or [200, 200, 200],
            ambient=color or [100, 100, 100],
            specular=[50, 50, 50]
        )

def get_file_path(filename: str) -> Path:
    base_path = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
    return base_path / filename

def reload_mesh_paths() -> dict:
    """
    Принудительная перезагрузка конфигурации путей.
    Полезно при изменении JSON во время работы приложения.
    """
    global obj_mesh
    obj_mesh = load_mesh_paths()
    return obj_mesh


def clear_cache():
    """Очистка кэшей генератора 3D"""
    import gc

    # 1. Очистка кэша trimesh
    try:
        if hasattr(trimesh, 'cache'):
            trimesh.cache.clear()
    except Exception as e:
        logger.warning(f"⚠️ Ошибка очистки trimesh.cache: {e}")

    # 2. 🔥 Очистка кэша загрузчиков мешей
    try:
        if hasattr(trimesh.exchange, 'load') and hasattr(trimesh.exchange.load, '_mesh_loaders'):
            trimesh.exchange.load._mesh_loaders.clear()
    except Exception as e:
        logger.warning(f"⚠️ Ошибка очистки mesh loaders: {e}")

    # 3. Перезагрузка конфигов + сборка мусора
    reload_mesh_paths()
    gc.collect()

def load_mesh_paths() -> dict:
    """
    Загружает пути к 3D-моделям из JSON-конфига.
    Возвращает словарь {class_name: relative_path}.
    """
    config_path = get_file_path('config/mesh_paths.json')

    if not config_path.exists():
        print(f"⚠️  Файл конфигурации не найден: {config_path}")
        print("📝  Используются пути по умолчанию (пустые)")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            mesh_paths = json.load(f)

        # Валидация: проверяем, что все значения — строки
        for key, value in mesh_paths.items():
            if not isinstance(value, str):
                print(f"⚠️  Неверный тип пути для '{key}': {type(value)}. Ожидалась строка.")
                mesh_paths[key] = ""

        print(f"✅  Загружено {len(mesh_paths)} путей к 3D-моделям из {config_path.name}")
        return mesh_paths

    except json.JSONDecodeError as e:
        print(f"❌  Ошибка парсинга JSON в {config_path}: {e}")
        print("📝  Используйте валидный JSON (проверьте запятые и кавычки)")
        return {}
    except Exception as e:
        print(f"❌  Ошибка загрузки конфигурации: {e}")
        return {}

obj_mesh = load_mesh_paths()

def _load_mesh_safe(path: Path, default_scale: float = 8.0, material_config=None):
    """Безопасная загрузка меша с валидацией."""
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh.is_empty:
            logger.error(f"❌ Пустой меш: {path}")
            # print(f"❌ Пустой меш: {path}")
            return None
        mesh.apply_scale(default_scale)
        mesh.fix_normals()
        # logger.info(f"✅ Загружен меш: {path.name}, вершин: {len(mesh.vertices)}")

        if material_config:
            mesh.visual = trimesh.visual.TextureVisuals(
                material=_create_pbr_material(**material_config)
            )

        print(f"✅ Загружен меш: {path.name}, вершин: {len(mesh.vertices)}")
        return mesh
    except Exception as e:
        # logger.error(f"❌ Ошибка загрузки {path}: {e}")
        print(f"❌ Ошибка загрузки {path}: {e}")
        return None

MATERIAL_CONFIG = {
    'Door': {'color': [139, 69, 19], 'roughness': 0.7},      # Дерево
    'Window': {'color': [173, 216, 230], 'metallic': 0.8},   # Стекло/металл
    'bathtube': {'color': [255, 250, 250], 'roughness': 0.3},# Керамика
    'cold_box': {'color': [192, 192, 192], 'metallic': 0.9}, # Металл
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
    floor.visual = trimesh.visual.TextureVisuals(
        material=_create_pbr_material(color=[150, 150, 250], roughness=0.9)
    )

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
            mesh_obj = _load_mesh_safe(get_file_path(obj_mesh[class_o]), 8, material_config=MATERIAL_CONFIG.get(class_o))

            for i, contour in enumerate(obj_contours[class_o]):
                # logger.info(")()()()(try copy mesh)()()()()(")
                mesh = mesh_obj.copy()

                contour_array = np.array(contour)

                width_d = contour_array[:, 0].max() - contour_array[:, 0].min()  # 1274 - 1175 = 99
                depth_d = contour_array[:, 1].max() - contour_array[:, 1].min()  # 1618 - 1467 = 151

                print(f"Габариты: {width_d} x {depth_d}")

                # Преобразуем контур в массив точек
                center_x = contour_array[:, 0].mean()  # Среднее по всем точкам
                center_y = contour_array[:, 1].mean()

                # Масштабируем координаты
                scaled_x = center_x * float(scale)
                scaled_y = center_y * float(scale)

                matrix_z_inversion = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]
                ])

                try:
                    min_z = mesh.bounds[0][2]
                    print(min_z)
                    # if min_z < 0:
                    #     matrix_z_inversion[:3, 3] = scaled_x, -(scaled_y), ((0.3-min_z))
                    # else:
                    matrix_z_inversion[:3, 3] = scaled_x, -(scaled_y), 0.3  # Устанавливаем смещение

                    rot = rotation_matrix(np.pi / 2, [1, 0, 0])
                    transform = concatenate_matrices(matrix_z_inversion, rot)
                    mesh.apply_transform(transform)

                    if width_d < depth_d:
                        # 1. Находим его центр (bounding box центроид)
                        center = mesh.bounding_box.centroid

                        # 2. Переносим в начало координат
                        mesh.apply_translation(-center)

                        # 3. Поворачиваем (например, на 45° вокруг оси Z)
                        rotation = rotation_matrix(np.pi / 2, [0, 0, 1])  # Ось Z
                        mesh.apply_transform(rotation)

                        # 4. Возвращаем на исходную позицию
                        mesh.apply_translation(center)

                    scene_objects.append(mesh)
                    print(f"Контур {class_o} #{i + 1} успешно преобразован в 3D-объект")

                except Exception as e:
                    print(f"Ошибка обработки контура {class_o} #{i + 1}: {str(e)}")
                    continue
    return scene_objects

def build_door(door_contours, wall_contours, scale, height = 2.7):
    scene_objects = []
    segments = extract_wall_segments_with_ids(wall_contours)
    mesh_door = _load_mesh_safe(get_file_path(obj_mesh['Door']), 1, material_config=MATERIAL_CONFIG.get('Door'))
    for i, contour in enumerate(door_contours):
        mesh = mesh_door.copy()

        contour_array = np.array(contour)

        width_d = contour_array[:, 0].max() - contour_array[:, 0].min()  # 1274 - 1175 = 99
        depth_d = contour_array[:, 1].max() - contour_array[:, 1].min()  # 1618 - 1467 = 151

        print(f"Габариты: {width_d} x {depth_d}")

        # Преобразуем контур в массив точек
        center_x = contour_array[:, 0].mean()  # Среднее по всем точкам
        center_y = contour_array[:, 1].mean()

        scaled_x = center_x * float(scale)
        scaled_y = center_y * float(scale)

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        try:
            rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
            transform = concatenate_matrices(matrix_z_inversion, rot)
            mesh.apply_transform(transform)

            if width_d < depth_d:
                rot_z = rotation_matrix(np.pi / 2, [0, 0, 1])
                mesh.apply_transform(rot_z)

                # 2. Замеряем текущие размеры модели (ТОЛЬКО после поворота, ДО масштабирования)
            model_w, model_d, model_h = mesh.extents

            # 3. Вычисляем масштаб по ДЛИННОЙ стороне
            long_contour = max(width_d, depth_d) * float(scale)
            long_model = max(model_w, model_d)
            scale_xy = long_contour / long_model  # Единый коэффициент для X и Y

            # 4. Масштабируем модель по X и Y (сохраняем пропорции)
            mesh.apply_scale([scale_xy, scale_xy, 1.0])

            # 5. Отдельно подгоняем высоту под параметр height
            # scale_z = height / mesh.extents[2]
            mesh.apply_scale([1.0, 1.0, 8])

            # 6. Перемещаем в нужную позицию (Z=0, чтобы дверь стояла на полу)
            mesh.apply_translation([scaled_x, -scaled_y, 0.0])

            # hight_door = mesh.extents.tolist()
            # # print(hight_door)
            # h_d_p = max(0.0, float(height) - hight_door[2])
            # if h_d_p > 1e-3:  # Создавать бокс только если есть зазор
            #     box = trimesh.primitives.Box(extents=[hight_door[0], hight_door[1], h_d_p])
            #     matrix_z_inversion[:3, 3] = scaled_x, -scaled_y, float(height) - h_d_p/2 # Устанавливаем смещение
            #     box.apply_transform(matrix_z_inversion)
            #     scene_objects.append(box)

            # Добавляем в сцену
            scene_objects.append(mesh)

            print(f"Контур DOOR #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура DOOR #{i + 1}: {str(e)}")
            continue
    return scene_objects

def build_window(window_position, wall_contours, scale, height = 2.7):
    scene_objects = []
    mesh_window = _load_mesh_safe(get_file_path(obj_mesh['Window']), 1, material_config=MATERIAL_CONFIG.get('Window'))
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


@timing
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