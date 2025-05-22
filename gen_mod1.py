import cv2
import numpy as np
import trimesh
from tabulate import tabulate

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

        # Масштабируем координаты
        scaled = contour_points * float(scale)

        # Замыкаем контур при необходимости
        if not np.allclose(scaled[0], scaled[-1]):
            scaled = np.vstack([scaled, scaled[0]])

        try:
            # Создаём 2D полигон
            polygon = trimesh.path.polygons.Polygon(scaled)

            # Экструдируем в 3D
            mesh = trimesh.creation.extrude_polygon(polygon, height=height)

            # Добавляем в сцену
            scene_objects.append(mesh)

            print(f"Контур #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура #{i + 1}: {str(e)}")
            continue

    if not scene_objects:
        raise ValueError("Не удалось создать ни одного 3D-объекта!")

    return trimesh.Scene(scene_objects)