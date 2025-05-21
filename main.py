import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from Cython.Shadow import py_int
from tabulate import tabulate

import img2wall as i2w


def create_3d_wall(start_point, end_point, height=3.0, thickness=0.2):
    """
    Создает 3D-модель стены между двумя точками
    :param start_point: Начальная точка стены (2D координаты)
    :param end_point: Конечная точка стены (2D координаты)
    :param height: Высота стены в метрах
    :param thickness: Толщина стены в метрах
    :return: 3D-меш стены (trimesh.Trimesh)
    """
    # Рассчитываем вектор направления и длину стены
    direction_vector = end_point - start_point
    wall_length = np.linalg.norm(direction_vector)

    # Рассчитываем угол поворота стены
    rotation_angle = np.arctan2(direction_vector[1], direction_vector[0])

    # Создаем матрицу преобразования
    transformation_matrix = trimesh.transformations.rotation_matrix(
        rotation_angle,
        [0, 0, 1]  # Ось вращения (Z-ось)
    )

    # Устанавливаем центр стены
    transformation_matrix[:2, 3] = (start_point + end_point) / 2

    # Создаем параллелепипед для стены
    wall_mesh = trimesh.creation.box(
        extents=[wall_length, thickness, height]
    )

    # Применяем трансформации
    wall_mesh.apply_transform(transformation_matrix)
    return wall_mesh


def build_3d_model(wall_contours, original_size, scale=0.1, height=3.0, thickness=0.2):
    """
    Строит 3D-модель помещения из контуров стен
    :param wall_contours: Контуры стен из обработки изображения
    :param original_size: Исходные размеры изображения (height, width)
    :param scale: Масштаб преобразования (пиксели в метры)
    :param height: Высота стен в метрах
    :param thickness: Толщина стен в метрах
    :return: Сцена с 3D-моделью (trimesh.Scene)
    """
    scene_objects = []

    # Обрабатываем все контуры
    for i, contour in enumerate(wall_contours):
        # Преобразование контура в массив точек
        contour_points = np.array(contour).reshape(-1, 2)

        contour_cv = contour.reshape(-1, 1, 2).astype(np.int32)

        # Упрощаем контур алгоритмом Рамера-Дугласа-Пьюкера
        epsilon_val = 0.0003 * cv2.arcLength(contour_cv, closed=True)
        approx_cv = cv2.approxPolyDP(contour_cv, epsilon_val, closed=True)
        contour_points = approx_cv.squeeze()

        # Пропускаем некорректные контуры
        if len(contour_points) < 2:
            print(f"Пропущен контур с {len(contour_points)} точками")
            continue

        print(f"\nКонтур #{i + 1}:")
        print(tabulate(
            contour_points,
            headers=['X', 'Y'],
            tablefmt="grid",
            showindex="always"
        ))

        # Масштабирование координат для текущего контура
        scaled_points = contour_points * scale

        # Отладочный вывод масштабированных точек
        print(f"\nМасштабированные точки контура #{i + 1}:")
        print(tabulate(
            scaled_points,
            headers=['X', 'Y'],
            tablefmt="fancy_grid",
            showindex="always"
        ))

        # Создаем стены между последовательными точками контура
        for j in range(len(scaled_points)):
            current_point = scaled_points[j]
            next_point = scaled_points[(j + 1) % len(scaled_points)]
            wall_length = np.linalg.norm(next_point - current_point)

            print(f"Точка {j}: длина стены = {wall_length:.2f} м")

            # Пропускаем слишком короткие стены
            while wall_length < 0.01:
                print(f"Пропущена стена нулевой длины между {current_point} и {next_point}")
                next_point = scaled_points[(j + 1) % len(scaled_points)]
                wall_length = np.linalg.norm(next_point - current_point)
                j = j+1
                print(f"Точка {j}: длина стены = {wall_length:.2f} м")
                # continue

            try:
                wall = create_3d_wall(
                    current_point,
                    next_point,
                    height=height,
                    thickness=thickness
                )
                scene_objects.append(wall)
            except Exception as e:
                print(f"Ошибка создания стены: {e}")
                continue

    if not scene_objects:
        raise ValueError("Не удалось создать ни одной 3D-стены!")

    return trimesh.Scene(scene_objects)

if __name__ == "__main__":
    try:
        # Обработка плана помещения
        wall_contours, image_size = i2w.process_floor_plan("test/0011.jpg")

        # Параметры моделирования
        MODEL_SCALE = 0.05 # 1 пиксель = 5 см
        WALL_HEIGHT = 20  # Высота потолков 2.7 метра
        WALL_THICKNESS = 0.001  # Толщина стен 20 см

        # Создание 3D-модели
        scene = build_3d_model(
            wall_contours,
            image_size,
            scale=MODEL_SCALE,
            height=WALL_HEIGHT,
            thickness=WALL_THICKNESS
        )

        # Визуализация результата
        scene.show()

        # Экспорт модели (опционально)
        # scene.export('model.glb')

    except FileNotFoundError as e:
        print(f"Ошибка загрузки файла: {e}")
    except ValueError as e:
        print(f"Ошибка обработки данных: {e}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")