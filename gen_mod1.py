import cv2
import numpy as np
import trimesh
from tabulate import tabulate
import img2wall as i2w
from trimesh.transformations import (translation_matrix,
                                     rotation_matrix,scale_matrix,
                                     concatenate_matrices)

mesh_door = trimesh.load('Door_Component.obj')

# def gen_floor(objects):
#     # 1. Получаем 2D проекции всех объектов
#     silhouette_meshes = []
#     for obj in objects:
#         section = obj.section(plane_origin=[0, 0, obj.bounds[0][2]], plane_normal=[0, 0, 1])
#         if section:
#             poly = section.to_planar()[0].polygons_full[0]
#             mesh = trimesh.creation.extrude_polygon(poly, height=0.001)
#             silhouette_meshes.append(mesh)
#
#     if silhouette_meshes:
#             combined = trimesh.boolean.union(silhouette_meshes)
#             # Получаем внешний полигон (объединенный контур)
#             union_polygon = combined.polygons_full[0]
#             print(combined)
#             print("++++++++++++++++++")
#             print(union_polygon)
#
#             # 4. Создаем 3D меш пола
#             floor = trimesh.creation.extrude_polygon(union_polygon, height=0.1)
#
#             # Позиционируем пол под всеми объектами
#             z_min = min([obj.bounds[0][2] for obj in objects])
#             floor.apply_translation([0, 0, z_min - 0.15])
#
#             # Настраиваем внешний вид
#             floor.visual.face_colors = [150, 150, 250, 200]  # Полупрозрачный синий
#
#             # Создаем сцену
#             # return floor
#             scene = trimesh.Scene(objects + [floor])
#             scene.show()
#
#     else:
#         print("Не удалось объединить силуэты")
#
#
#     # if silhouettes:
#     #     # Собираем все полигоны в один Path2D
#     #     combined = trimesh.path.util.concatenate(silhouettes)
#     #
#     #     # Получаем внешний контур (объединение всех полигонов)
#     #     polygon = combined.polygons_full[0]
#     #
#     #     # 3. Создаем плоский меш из полигона
#     #     floor = trimesh.creation.extrude_polygon(polygon, height=0.05)
#     #
#     #     # 4. Размещаем пол под всеми объектами
#     #     z_min = min([obj.bounds[0][2] for obj in objects])
#     #     floor.apply_translation([0, 0, z_min - 0.1])  # чуть ниже самого нижнего объекта
#     #
#     #     # Задаем цвет полу (например, серый)
#     #     floor.visual.face_colors = [200, 200, 200, 255]
#     #
#     #     # Создаем сцену
#     #     return floor
#     #     # scene = trimesh.Scene(objects + [floor])
#     #     # scene.show()
#     # else:
#     #     print("Не удалось создать силуэты для объектов")

def build_door(wall_contours, scale):
    scene_objects = []
    for i, contour in enumerate(wall_contours):
        mesh = mesh_door.copy()
        wight_d = contour[2][0] - contour[0][0]
        hight_d = contour[1][1] - contour[0][1]
        contour = i2w.average_close_points(contour, 300)
        contour = np.column_stack((contour, np.zeros(len(contour))))

        print(contour)
        # Преобразуем контур в массив точек
        contour_points = np.array(contour[0])

        # Масштабируем координаты
        scaled = contour_points * float(scale)
        print(scaled)
        print(scaled[0])
        print(translation_matrix(scaled))

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        try:
            matrix_z_inversion[:3, 3] = scaled[0], -scaled[1], 0  # Устанавливаем смещение
            # Создаём 2D полигон
            mesh.apply_scale(6)
            rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
            transform = concatenate_matrices(matrix_z_inversion, rot)
            mesh.apply_transform(transform)
            print(transform)
            if wight_d < hight_d:
                # 1. Находим его центр (bounding box центроид)
                center = mesh.bounding_box.centroid

                # 2. Переносим в начало координат
                mesh.apply_translation(-center)

                # 3. Поворачиваем (например, на 45° вокруг оси Z)
                angle = np.pi / 2  # 90° в радианах
                rotation = rotation_matrix(angle, [0, 0, 1])  # Ось Z
                mesh.apply_transform(rotation)

                # 4. Возвращаем на исходную позицию
                mesh.apply_translation(center)

            # Добавляем в сцену
            scene_objects.append(mesh)

            print(f"Контур #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура DOOR #{i + 1}: {str(e)}")
            continue

    return scene_objects

def build_window():
    print("window")

def build_3d_model(wall_contours, original_size, scale=0.1, height=3.0, esp = 0.00025):
    """
    Строит 3D-модель помещения экструдированием контуров стен
    :param wall_contours: Контуры стен из обработки изображения
    :param original_size: Исходные размеры изображения (height, width)
    :param scale: Масштаб преобразования (пиксели в метры)
    :param height: Высота помещения в метрах
    :param esp: округление стен
    :return: Сцена с 3D-моделью (trimesh.Scene)
    """

    scene_objects = []

    for i, contour in enumerate(wall_contours):
        # Преобразуем контур в массив точек
        contour_points = np.array(contour).reshape(-1, 2)

        # Упрощаем контур
        contour_cv = contour_points.reshape(-1, 1, 2).astype(np.int32)
        epsilon = float(esp) * cv2.arcLength(contour_cv, closed=True)
        approx = cv2.approxPolyDP(contour_cv, epsilon, closed=True)
        contour_points = approx.squeeze()

        # Пропускаем контуры с малым количеством точек
        if len(contour_points) < 3:
            print(f"Контур #{i + 1} пропущен (мало точек: {len(contour_points)})")
            continue

        # next_wall = wall_contours[(i + 1) % len(wall_contours)]

        # build_door(wall_contours[i], next_wall, scale)

        # Масштабируем координаты
        scaled = contour_points * float(scale)

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
            print("стена = ",scaled)

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

    # scene_objects.append(gen_floor(scene_objects))

    return trimesh.Scene(scene_objects)