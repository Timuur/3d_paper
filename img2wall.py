import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

from decorator import append
from tabulate import tabulate


def sotr_cor(ar_cor, out_img):
    lines = []
    line = []
    visited = set()
    points = ar_cor
    # prev_i = len(ar_cor) - 1
    # prev_i1 = ar_cor[-2]
    # for i in range(0, len(ar_cor)):
    #     x, y = ar_cor[i].ravel()
    #     for i1 in range(len(ar_cor)):
    #         x1, y1 = ar_cor[i1].ravel()
    #         if x != x1 and y != y1:
    #             if i % 2 == 1:
    #                 if (y == y1) or(y == (y1-1)) or (y == (y1+1)):
    #                     cv2.line(out_img, (x, y), (x1, y1), (0, 255, 0), 3)
    #                     line = x,y,x1,y1
    #                     break
    #             else:
    #                 if (x == x1) or (x == (x1 - 1)) or (x == (x1 + 1)):
    #                     cv2.line(out_img, (x, y), (x1, y1), (0, 0, 255), 2)
    #                     line = x,y,x1,y1
    #                     break
    #         lines.append(line)
    #
    #     prev_i = i

    def find_cycle(current, prev_coord='y'):
        # if len(lines) == len(points) and lines[0] == current:
        if lines[0] == current:
            return lines
        if current in visited:
            return None
        visited.add(current)
        x, y = current
        next_coord = 'x' if prev_coord == 'y' else 'y'
        for point in points:
            px, py = point
            if next_coord == 'x' and abs(px - x) == 1:
                result = find_cycle(point, 'x')
                if result:
                    lines.append(point)
                    return lines
            elif next_coord == 'y' and abs(py - y) == 1:
                result = find_cycle(point, 'y')
                if result:
                    lines.append(point)
                    return lines
        visited.remove(current)
        return None

    print(lines)

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

def detect_walls_by_color(image):
    """
    Создает маску для стен на основе цветового диапазона в HSV
    :param image: Входное изображение BGR
    :return: Бинарная маска стен
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Диапазон цветов для стен (настройте под ваш случай)
    lower_color = np.array([11, 146, 0])
    upper_color = np.array([255, 255, 255])

    return cv2.inRange(hsv_image, lower_color, upper_color)

def detect_and_draw_lines(image, output_image):
    """
    Обнаруживает и рисует линии на изображении
    :param image: Исходное изображение для обработки
    :param output_image: Изображение для визуализации результатов
    """
    # Детектирование краев
    edges = cv2.Canny(image, 50, 150)

    # Параметры детекции линий:
    lines = cv2.HoughLinesP(
        edges,
        1,  # Разрешение rho (пиксели)
        np.pi / 150,  # Разрешение theta (радианы)
        threshold=10,  # Минимальное количество пересечений
        minLineLength=5,  # Минимальная длина линии
        maxLineGap=1  # Максимальный разрыв между линиями
    )

    # Отрисовка линий
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return lines

def detect_and_draw_squar(image, output_image):
    """
    Обнаруживает и рисует линии на изображении
    :param image: Исходное изображение для обработки
    :param output_image: Изображение для визуализации результатов
    """
    # Детектирование краев
    edges = cv2.Canny(image, 50, 150)

    # Параметры детекции линий:
    lines = cv2.HoughLinesP(
        edges,
        1,  # Разрешение rho (пиксели)
        np.pi / 160,  # Разрешение theta (радианы)
        threshold=20,  # Минимальное количество пересечений
        minLineLength=5,  # Минимальная длина линии
        maxLineGap=1  # Максимальный разрыв между линиями
    )

    # Отрисовка линий
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return lines

def detect_corner(image, output_image):
    corners = cv2.goodFeaturesToTrack(image, 300, .06, 5)
    corners = np.intp(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(output_image, (x, y), 3, (0, 0, 255), -1)  # Используем красный цвет (в RGB)
    return corners


def replace_gray_in_monochrome(image, lower_gray=50, upper_gray=230, target_value=255):
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

    processed_image = replace_gray_in_monochrome(cleaned_image)
    processed_image = replace_gray_in_monochrome(processed_image, 0, 10, 0)

    # _________________________________________________________________________________________
    corners = detect_corner(cleaned_image, bordered_image)

    # line1 = detect_and_draw_lines(processed_image, original_image)
    # line2 = detect_and_draw_squar(processed_image, original_image)

    # Поиск контуров
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров рамки
    filtered_contours = []
    border_margin = -10
    image_height, image_width = processed_image.shape
    #_________________________________________________________________________________________


    # _________________________________________________________________________________________

    for i, contour in enumerate(contours):
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
