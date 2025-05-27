import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

from tabulate import tabulate

def average_close_points(points, threshold=2):
    """
    Усредняет близкие точки в массиве кортежей (x, y).
    Точки считаются близкими, если расстояние между ними <= threshold.
    Возвращает список усредненных точек.
    """
    n = len(points)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if not visited[i]:
            cluster = []
            queue = [i]
            visited[i] = True
            while queue:
                current = queue.pop(0)
                cluster.append(points[current])
                # Поиск всех соседей текущей точки
                for j in range(n):
                    if not visited[j]:
                        dx = points[current][0] - points[j][0]
                        dy = points[current][1] - points[j][1]
                        if dx ** 2 + dy ** 2 <= threshold ** 2:
                            visited[j] = True
                            queue.append(j)
            clusters.append(cluster)

    # Усреднение кластеров
    averaged_points = []
    for cluster in clusters:
        if cluster:
            avg_x = sum(x for x, y in cluster) / len(cluster)
            avg_y = sum(y for x, y in cluster) / len(cluster)
            averaged_points.append((round(avg_x), round(avg_y)))

    return averaged_points

def average_close_points_X(points, threshold=2):
    """
    Усредняет близкие точки в массиве кортежей (x, y).
    Точки считаются близкими, если расстояние между ними <= threshold.
    Возвращает список усредненных точек.
    """
    n = len(points)

    for i in range(n - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Усредняем X, если разница в пределах порога
        if dx <= threshold:
            avg_x = (x1 + x2 + 1) // 2
            points[i][0] = avg_x
            points[i + 1][0] = avg_x

        # Усредняем Y, если разница в пределах порога
        if dy <= threshold:
            avg_y = (y1 + y2 + 1) // 2
            points[i][1] = avg_y
            points[i + 1][1] = avg_y

    return points

def average_close_points_XX(points, threshold=2):
    """
    Усредняет близкие точки в массиве кортежей (x, y).
    Точки считаются близкими, если расстояние между ними <= threshold.
    Возвращает список усредненных точек.
    """
    n = len(points)
    # visited = [] * n
    for i in range(n-1):
        # x1, y1 = points[i].ravel()
        # x2, y2 = points[i+1].ravel()
        n1 = len(points[i])
        # visited[i] = [False] * n1
        n2 = len(points[(i + 1) % n])
        for j in range(n1-1):
            x1, y1 = points[i][j].ravel()
            for j1 in range(n2 - 1):
                x2, y2 = points[(i + 1) % n][j1].ravel()

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Усредняем X, если разница в пределах порога
                if dx <= threshold:
                    avg_x = (x1 + x2 + 1) // 2
                    points[i][j] = avg_x, y1
                    points[(i + 1) % n][j1]= avg_x, y2

                # Усредняем Y, если разница в пределах порога
                if dy <= threshold:
                    avg_y = (y1 + y2 + 1) // 2
                    points[i][j] = x1, avg_y
                    points[(i + 1) % n][j1] = x2, avg_y

    return points

# def average_close_coordinates_for_array(
#     contours: list[cv2.Mat | np.ndarray],
#     threshold: int = 1,
#     count: int = 1
# ) -> list[np.ndarray]:
#     """
#     Обрабатывает последовательность контуров (cv2.Mat или np.ndarray),
#     усредняя близкие координаты X и Y для каждой точки всех контуров.
#
#     Параметры:
#         contours: Список контуров в формате (N, 1, 2) или (N, 2).
#         threshold: Максимальная разница для усреднения координат.
#
#     Возвращает:
#         Список контуров в формате np.ndarray с обновленными координатами.
#     """
#     processed_contours = []
#     for contour in contours:
#         contour_array = np.array(contour) if isinstance(contour, cv2.Mat) else contour
#         original_dtype = contour_array.dtype  # Сохраняем исходный тип данных
#
#         # Извлекаем точки и преобразуем в список кортежей
#         points = contour_array.reshape(-1, 2)
#         tuple_points = [tuple(map(int, p)) for p in points]  # Гарантируем целые числа
#
#         # Усреднение координат (обновленная функция)
#         smoothed_points = _average_close_coordinates(tuple_points, threshold,count)
#
#         # Преобразуем обратно в исходный формат и тип данных
#         reshaped_points = np.array(smoothed_points, dtype=original_dtype)
#         if contour_array.ndim == 3:
#             reshaped_points = reshaped_points.reshape(-1, 1, 2)
#         processed_contours.append(reshaped_points)
#     return processed_contours
#
# def _average_close_coordinates(
#         points: list[tuple[int, int]],
#         threshold: int,
#         count: int
# ) -> list[tuple[int, int]]:
#     points_list = [list(p) for p in points]  # Работаем с изменяемым списком
#
#     # Кластеризация по X и Y отдельно
#     def cluster_and_average(coord_idx: int):
#         parent = list(range(len(points_list)))
#
#         def find(u):
#             while parent[u] != u:
#                 parent[u] = parent[parent[u]]
#                 u = parent[u]
#             return u
#
#         # Объединяем точки с близкими координатами
#         for i in range(len(points_list)):
#             for j in range(i + 1, len(points_list)):
#                 if abs(points_list[i][coord_idx] - points_list[j][coord_idx]) <= threshold:
#                     root_i, root_j = find(i), find(j)
#                     if root_i != root_j:
#                         parent[root_j] = root_i
#
#         # Усредняем координаты в кластерах
#         clusters = {}
#         for idx in range(len(points_list)):
#             root = find(idx)
#             clusters.setdefault(root, []).append(idx)
#
#         for cluster in clusters.values():
#             if len(cluster) > 1:
#                 avg = round(sum(points_list[i][coord_idx] for i in cluster) / len(cluster))
#                 for i in cluster:
#                     points_list[i][coord_idx] = int(avg)  # Гарантируем целое число
#
#     # Обрабатываем X (0) и Y (1)
#     for u in range(count):
#         cluster_and_average(0)
#         cluster_and_average(1)
#
#     return [tuple(p) for p in points_list]


# def average_cls_points_x(contour: Sequence[cv2.Mat | np.ndarray], threshold=2):
#     # Ensure the input is a NumPy array
#     # contour = np.asarray(contour)
#     # original_shape = contour.shape
#     contour_reshaped = contour.reshape(-1, 2)
#     # contour_list = contour_reshaped.tolist()
#     n = len(contour_reshaped)
#
#     def process_coordinate(coord_idx):
#         parent = list(range(n))
#
#         def find(u):
#             while parent[u] != u:
#                 parent[u] = parent[parent[u]]  # Path compression
#                 u = parent[u]
#             return u
#
#         # Group indices with coordinates within the threshold
#         for i in range(n):
#             for j in range(i + 1, n):
#                 if abs(contour_reshaped[i][coord_idx] - contour_reshaped[j][coord_idx]) <= threshold:
#                     root_i = find(i)
#                     root_j = find(j)
#                     if root_i != root_j:
#                         parent[root_j] = root_i  # Union
#
#         # Compute averages for each cluster
#         clusters = {}
#         for i in range(n):
#             root = find(i)
#             clusters.setdefault(root, []).append(i)
#
#         for indices in clusters.values():
#             if len(indices) > 1:
#                 avg = round(sum(contour_reshaped[i][coord_idx] for i in indices) / len(indices))
#                 for i in indices:
#                     contour_reshaped[i][coord_idx] = avg
#
#     # Process X and Y coordinates
#     process_coordinate(0)  # X-axis
#     process_coordinate(1)  # Y-axis
#
#     # Convert back to original shape
#     processed_contour = np.array(contour_reshaped, dtype=np.int32)
#     return processed_contour

def add_white_border(image, border_size=20):
    """
    Добавляет белую рамку вокруг изображения
    :param image: Входное изображение BGR
    :param border_size: Толщина рамки в пикселях
    :return: Изображение с белой рамкой
    """
    return cv2.copyMakeBorder(
        image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

def apply_adaptive_threshold(gray_image):
    """
    Применяет адаптивную пороговую обработку для бинаризации изображения
    :param gray_image: Одноканальное изображение в градациях серого
    :return: Бинаризованное изображение
    """
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Размер соседней области
        2  # Константа для вычитания
    )

# def detect_walls_by_color(image):
#     """
#     Создает маску для стен на основе цветового диапазона в HSV
#     :param image: Входное изображение BGR
#     :return: Бинарная маска стен
#     """
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     # Диапазон цветов для стен (настройте под ваш случай)
#     lower_color = np.array([11, 146, 0])
#     upper_color = np.array([255, 255, 255])
#
#     return cv2.inRange(hsv_image, lower_color, upper_color)
#
# def detect_and_draw_lines(image, output_image):
#     """
#     Обнаруживает и рисует линии на изображении
#     :param image: Исходное изображение для обработки
#     :param output_image: Изображение для визуализации результатов
#     """
#     # Детектирование краев
#     edges = cv2.Canny(image, 50, 150)
#
#     # Параметры детекции линий:
#     lines = cv2.HoughLinesP(
#         edges,
#         1,  # Разрешение rho (пиксели)
#         np.pi / 150,  # Разрешение theta (радианы)
#         threshold=10,  # Минимальное количество пересечений
#         minLineLength=5,  # Минимальная длина линии
#         maxLineGap=1  # Максимальный разрыв между линиями
#     )
#
#     # Отрисовка линий
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             # cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     return lines
#
# def detect_and_draw_squar(image, output_image):
#     """
#     Обнаруживает и рисует линии на изображении
#     :param image: Исходное изображение для обработки
#     :param output_image: Изображение для визуализации результатов
#     """
#     # Детектирование краев
#     edges = cv2.Canny(image, 50, 150)
#
#     # Параметры детекции линий:
#     lines = cv2.HoughLinesP(
#         edges,
#         1,  # Разрешение rho (пиксели)
#         np.pi / 160,  # Разрешение theta (радианы)
#         threshold=20,  # Минимальное количество пересечений
#         minLineLength=5,  # Минимальная длина линии
#         maxLineGap=1  # Максимальный разрыв между линиями
#     )
#
#     # Отрисовка линий
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(output_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
#     return lines

# def detect_corner(image, output_image):
#     corners = cv2.goodFeaturesToTrack(image, 900, .06, 2)
#     corners = np.intp(corners)
#
#     for i in corners:
#         x, y = i.ravel()
#         cv2.circle(output_image, (x, y), 1, (0, 0, 255), -1)  # Используем красный цвет (в RGB)
#     return corners

def replace_gray_in_monochrome(image, lower_gray=39, upper_gray=255, target_value=255):
    """
    Заменяет диапазон серых пикселей в монохромном изображении на белые.

    Параметры:
    - image: одноканальное изображение (numpy array).
    - lower_gray: нижняя граница серого (0-255).
    - upper_gray: верхняя граница серого (0-255).
    - target_value: целевое значение (по умолчанию 255 — белый).
    """
    # Создаем маску пикселей в заданном диапазоне
    mask = (image >= lower_gray) & (image <= upper_gray)
    # Заменяем пиксели на целевое значение
    image[mask] = target_value
    return image

def process_floor_plan(image_path, border_size=10):
    """
    Основная функция обработки плана помещения
    :param image_path: Путь к исходному изображению
    :param border_size: Размер обрамляющей рамки
    :return: Контуры стен и исходные размеры изображения
    """
    # _________________________________________________________________________________________
    # Загрузка изображения и предобработка
    original_image = cv2.imread(image_path)

    # Замена черного фона на белый
    # black_pixels_mask = np.all(original_image[:, :, :3] == [0, 0, 0], axis=-1)
    # original_image[black_pixels_mask, :3] = [255, 255, 255]

    # Сохранение исходных размеров
    original_height, original_width = original_image.shape[:2]

    # Добавление белой рамки для обработки краев
    bordered_image = add_white_border(original_image, border_size)

    # Сегментация стен по цвету
    # wall_mask = detect_walls_by_color(bordered_image)

    # Преобразование маски в 3-канальное изображение
    # wall_mask_bgr = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)

    # Бинаризация с адаптивным порогом
    gray_image = cv2.cvtColor(bordered_image, cv2.COLOR_BGR2GRAY)
    binary_image = apply_adaptive_threshold(gray_image)

    # Морфологическая обработка для устранения шума
    kernel = np.ones((10, 10), np.uint8)
    cleaned_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Детекция и отрисовка линий
    # line1 = detect_and_draw_lines(cleaned_image, bordered_image)

    processed_image = replace_gray_in_monochrome(cleaned_image, 35, 255, 255)
    processed_image = replace_gray_in_monochrome(processed_image, 0, 34, 0)

    # _________________________________________________________________________________________
    # corners = detect_corner(cleaned_image, bordered_image)

    # line1 = detect_and_draw_lines(processed_image, original_image)
    # line2 = detect_and_draw_squar(processed_image, original_image)

    # Поиск контуров
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров рамки
    filtered_contours = []
    border_margin = -10
    image_height, image_width = processed_image.shape

    # contours = average_close_coordinates_for_array(contours, 4, 4)

    # contours = average_close_coordinates_for_array(contours, 2)

    #_________________________________________________________________________________________
    # print(corners)
    #
    # print(contours)
    #
    # # Исходные данные (пример)
    # first_array = np.array(corners)  # Первый массив точек
    # sub_arrays = contours  # Кортеж подмассивов из второго массива
    #
    # all_second_points = []
    # for i, sub in enumerate(sub_arrays):
    #     for point in sub:
    #         x1, y1, _, _ = cv2.boundingRect(point)  # Доступ к координатам через [0][0] и [0][1]
    #         all_second_points.append((x1, y1, i, sub))
    #
    # # Корректное создание множеств для сравнения
    # first_points_set = set((p[0][0], p[0][1]) for p in first_array)
    # second_points_set = set((x, y) for x, y, _, _ in all_second_points)
    #
    # # Поиск отсутствующих точек с сохранением 3D-структуры
    # missing_points = [np.array([[[x, y]]]) for x, y in first_points_set
    #                   if (x, y) not in second_points_set]
    #
    # # Функция поиска ближайшей точки
    # def find_closest(target, points):
    #     min_dist = float('inf')
    #     closest = None
    #     for p in points:
    #         d = distance.euclidean(target, (p[0], p[1]))
    #         if d < min_dist:
    #             min_dist = d
    #             closest = p
    #     return closest
    #
    # # Вставка точек с учётом 3D-структуры
    # for mp in missing_points:
    #     x_val, y_val = mp[0][0][0], mp[0][0][1]  # Доступ через три уровня индексов
    #     closest_info = find_closest((x_val, y_val), all_second_points)
    #
    #     if closest_info:
    #         closest_x, closest_y, sub_idx, sub = closest_info
    #         # Поиск позиции с учётом 3D-структуры
    #         position = np.where((sub == [[[closest_x, closest_y]]]).all(axis=2))[0][0]
    #
    #         # Создание новой точки с правильной размерностью (1,1,2)
    #         new_point = np.array([[[x_val, y_val]]], dtype=sub.dtype)
    #
    #         # Вставка с сохранением размерности
    #         updated_sub = np.insert(sub, position + 1, new_point, axis=0)
    #
    #         # Обновление подмассивов
    #         sub_arrays = list(sub_arrays)
    #         sub_arrays[sub_idx] = updated_sub
    #         sub_arrays = tuple(sub_arrays)
    #
    # print(sub_arrays)
    # contours = sub_arrays
    # _________________________________________________________________________________________

    for i, contour in enumerate(contours):
        contour = np.array( average_close_points(   contour.reshape(-1, 2), 3))
        contour = np.array( average_close_points_X(   contour.reshape(-1, 2), 4))
        # contour = np.array( average_close_points_X(   contour.reshape(-1, 2), 2))
        # contour = np.array( average_close_points_X(   contour.reshape(-1, 2), 1))
        # print(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (x > border_size + border_margin
                and y > border_size + border_margin
                and (x + w) < (image_width - border_size - border_margin)
                and (y + h) < (image_height - border_size - border_margin)):
            filtered_contours.append(contour)
            cv2.drawContours(bordered_image, [contour], 0, (randrange(50,256), randrange(50, 256), randrange(50, 256)), 1)
        contour_points = contour.reshape(-1, 2)

        # Вывод таблицы для каждого контура
        print(f"\nКонтур #{i + 1}:")
        print(tabulate(
            contour_points,
            headers=['X', 'Y'],
            tablefmt="grid",
            showindex="always"
        ))

        # contour_points = average_close_points_X(contour_points, 4)
        # contour_points = average_close_points_X(contour_points, 2)
        # contour_points = average_close_points_X(contour_points, 1)
        # # Вывод таблицы для каждого контура
        # print(f"\nКонтур #{i + 1}:")
        # print(tabulate(
        #     contour_points,
        #     headers=['X', 'Y'],
        #     tablefmt="grid",
        #     showindex="always"
        # ))
    # _________________________________________________________________________________________
    # Визуализация промежуточных результатов
    debug_images = [
        original_image,
        bordered_image,
        gray_image,
        binary_image,
        cleaned_image,
        processed_image
    ]

    # Конвертация BGR в RGB для корректного отображения
    debug_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                        for img in debug_images]

    titles = [
        'Original Image',
        'Bordered Image',
        'Grayscale',
        'Binary Image',
        'Cleaned Image',
        'processed Image'
    ]

    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(debug_images_rgb, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap='gray' if i > 1 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    # _________________________________________________________________________________________
    #cv2.imwrite('output.png', cleaned_image)

    if not filtered_contours:
        raise ValueError("Не обнаружено стен. Проверьте параметры обработки изображения.")

    # Корректировка координат контуров (удаление рамки)
    wall_contours = [cnt - border_size for cnt in filtered_contours]

    # return sorted_corner, (original_height, original_width)
    return wall_contours, (original_height, original_width)
