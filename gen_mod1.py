import numpy as np
import trimesh
import img2wall as i2w
from scipy.spatial import ConvexHull
from trimesh.transformations import rotation_matrix, concatenate_matrices
from shapely.geometry import Polygon

mesh_door = trimesh.load('Door_Component.obj')
mesh_window = trimesh.load('window1.obj')

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


def build_door(door_contours, wall_contours, scale, height = 2.7):
    scene_objects = []
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
                mesh.apply_translation(center)

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

def build_3d_model(wall_contours, original_size, scale=0.1, height=3.0):
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