import cv2
import numpy as np
import trimesh
from tabulate import tabulate

def build_door(wall1, wall2):
    box = []
    for i1, point1 in enumerate(wall1):
        x1,y1 = point1.ravel()
        for i2, point2 in enumerate(wall2):
            x2, y2 = point2.ravel()
            wall_length = np.linalg.norm(point1 - point2)
            if ((x1 == x2) or (x1 == x2 -1) or (x1 == x2+1)) and 150<abs(wall_length)<199:
                print(f"Точка {i1} =[{x1, y1}] b Точка{i2} =[{x2, y2 }]: длина стены = {wall_length:.2f} м")
                box.append(point1)
                box.append(point2)
            else:
                if ((y1 == y2) or (y1 == y2 -1) or (y1 == y2+1)) and 150<abs(wall_length)<199:
                    print(f"Точка {i1} =[{x1, y1}] b Точка{i2} =[{x2, y2 }]: длина стены = {wall_length:.2f} м")
                    box.append(point1)
                    box.append(point2)

    print("door")
    print(box)

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

        next_wall = wall_contours[(i + 1) % len(wall_contours)]

        build_door(wall_contours[i], next_wall)

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

    return trimesh.Scene(scene_objects)