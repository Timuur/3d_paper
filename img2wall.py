import cv2
import numpy as np
from random import randrange

from matplotlib import pyplot as plt

from check_img_ai import get_coord

classes_dict = {'Door': [],
            'GasPlate': [],
            'Wardor': [],
            'Wall': [],
            'Window': [],
            'bathtube': [],
            'box': [],
            'door_bath': [],
            'cold_box': [],
            'door-s': [],
            'door_l': [],
            'door_balcon': [],
            'door_bath': [],
            'h-wall': [],
            'door_vhod_l': [],
            'sink': [],
            'sink_kitchen': [],
            'balcon_wall': [],
            'toulet': [],
            'wash_machine': [],
            'win_in_wall': []}

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

def average_close2points(points, threshold=2):
    """
    Усредняет близкие точки в списке numpy-массивов.
    Работает с массивами вида (N, 1, 2) или (N, 2).
    """
    points = [arr.copy() for arr in points]  # Копируем, чтобы не менять исходные данные

    for i in range(len(points)):
        for j in range(len(points[i])):
            # Получаем координаты точки, учитывая возможную структуру массива
            if points[i][j].ndim == 2:  # Если массив вида [[x, y]]
                x1, y1 = points[i][j][0]
            else:  # Если массив вида [x, y] или просто скаляр (но в вашем случае это массив)
                x1, y1 = points[i][j].ravel()

            # Сравниваем со всеми остальными точками
            for k in range(len(points)):
                # for k in range(i, len(points)):
                for l in range(len(points[k])):
                    # for l in range(j + 1 if k == i else 0, len(points[k])):
                    # Аналогично получаем x2, y2
                    if points[k][l].ndim == 2:
                        x2, y2 = points[k][l][0]
                    else:
                        x2, y2 = points[k][l].ravel()

                    # Если точки близки, усредняем их
                    if abs(x1 - x2) <= threshold:
                        avg_x = (x1 + x2) // 2

                        # Обновляем координаты, сохраняя структуру массива
                        if points[i][j].ndim == 2:
                            points[i][j][0][0] = avg_x
                        else:
                            points[i][j][:][0] = avg_x

                        if points[k][l].ndim == 2:
                            points[k][l][0][0] = avg_x
                        else:
                            points[k][l][:][0] = avg_x
                    # Если точки близки, усредняем их
                    if abs(y1 - y2) <= threshold:
                        avg_y = (y1 + y2) // 2

                        # Обновляем координаты, сохраняя структуру массива
                        if points[i][j].ndim == 2:
                            points[i][j][0][1] = avg_y
                        else:
                            points[i][j][:][1] = avg_y

                        if points[k][l].ndim == 2:
                            points[k][l][0][1] = avg_y
                        else:
                            points[k][l][:][1] = avg_y
    return points

#_______________________________________________________________________________________________________________________

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

#______________________________________________________________________________________________________________________

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

def get_scale(classes_dict, original_image, pixel_on_map=151):
    #1 получить координаты дверей
    #2 зная примерно идеальную длину двери высчитать необходимый масштаб
    #3 применить масштаб
    for i, contour in enumerate(classes_dict['Door']):
        wight_d_i = contour[2][0] - contour[0][0]
        hight_d_i = contour[1][1] - contour[0][1]
        if wight_d_i > hight_d_i:
            # scale_win = wight_d_i * float(scale) / hight_win[0]
            scale_p = pixel_on_map / wight_d_i
            print('(', scale_p,")hight_d = ", pixel_on_map, ' / ', wight_d_i,'/', wight_d_i*0.05)
        else:
            # scale_win = hight_d_i * float(scale) / hight_win[0]
            scale_p =  pixel_on_map / hight_d_i
            print('(', scale_p,")hight_d = ", pixel_on_map, '/', hight_d_i,'/', hight_d_i*0.05)
    # for i, contour in enumerate(classes_dict['door_bath']):
    #     wight_d_i = contour[2][0] - contour[0][0]
    #     hight_d_i = contour[1][1] - contour[0][1]
    #     if wight_d_i > hight_d_i:
    #         # scale_win = wight_d_i * float(scale) / hight_win[0]
    #         scale_p = pixel_on_map / wight_d_i
    #         print('(', scale_p,")hight_d = ", pixel_on_map, '/', wight_d_i)
    #     else:
    #         # scale_win = hight_d_i * float(scale) / hight_win[0]
    #         scale_p =  pixel_on_map / hight_d_i
    #         print('(', scale_p,")hight_d = ", pixel_on_map, '/', hight_d_i)
    # for i, contour in enumerate(classes_dict['door_bath_l']):
    #     wight_d_i = contour[2][0] - contour[0][0]
    #     hight_d_i = contour[1][1] - contour[0][1]
    #     if wight_d_i > hight_d_i:
    #         # scale_win = wight_d_i * float(scale) / hight_win[0]
    #         scale_p = pixel_on_map / wight_d_i
    #         print('(', scale_p,")hight_d = ", pixel_on_map, '/', wight_d_i)
    #     else:
    #         # scale_win = hight_d_i * float(scale) / hight_win[0]
    #         scale_p =  pixel_on_map / hight_d_i
    #         print('(', scale_p,")hight_d = ", pixel_on_map, '/', hight_d_i)

def process_floor_plan(image_path, border_size=20):
    """
    Основная функция обработки плана помещения
    :param image_path: Путь к исходному изображению
    :param border_size: Размер обрамляющей рамки
    :return: Контуры стен и исходные размеры изображения
    """
    # _________________________________________________________________________________________
    # Загрузка изображения и предобработка
    original_image = cv2.imread(image_path)


    # _________________________________________________________________________________________
    # _________________________________________________________________________________________

    # class_path_txt = get_file_path('ai_model/classes.txt')


    detections, labels = get_coord(image_path)
    # print(detections)
    # print(labels)
    for i in range(len(detections)):
        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int)  # Extract individual coordinates and convert to int
        xy1 = xmin + border_size, ymin + border_size
        xy2 = xmin + border_size, ymax + border_size
        xy4 = xmax + border_size, ymin + border_size
        xy3 = xmax + border_size, ymax + border_size
        xyy = xy1,xy2,xy3,xy4

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        # conf = detections[i].conf.item()

        if classname in classes_dict:
            classes_dict[classname].append(np.array(xyy))

    get_scale(classes_dict,original_image)
    # _________________________________________________________________________________________
    # _________________________________________________________________________________________


    # Замена черного фона на белый
    # black_pixels_mask = np.all(original_image[:, :, :3] == [0, 0, 0], axis=-1)
    # original_image[black_pixels_mask, :3] = [255, 255, 255]

    # Сохранение исходных размеров
    original_height, original_width = original_image.shape[:2]

    # Добавление белой рамки для обработки краев
    bordered_image = add_white_border(original_image, border_size)

    # Бинаризация с адаптивным порогом
    gray_image = cv2.cvtColor(bordered_image, cv2.COLOR_BGR2GRAY)
    binary_image = apply_adaptive_threshold(gray_image)

    # Морфологическая обработка для устранения шума
    kernel = np.ones((10, 10), np.uint8)
    cleaned_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    processed_image = replace_gray_in_monochrome(cleaned_image, 35, 255, 255)
    processed_image = replace_gray_in_monochrome(processed_image, 0, 34, 0)

    # _________________________________________________________________________________________

    # Поиск контуров
    contours_wall, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for i in contours_wall:
    #     x, y = i.ravel()
    #     cv2.circle(processed_image, (x, y), 3, (255, 0, 0), -1)  # Красный цвет в RGB

    # Фильтрация контуров рамки
    filtered_contours_wall = []
    border_margin = -10
    image_height, image_width = processed_image.shape



    # print(f"\nКонтур #no abs:")
    # print(contours)
    # print("________________________________________________________________________")


    # print(f"\nКонтур #abs:")
    # print(contours)
    # print("________________________________________________________________________")


    # debug_box = [window,box,door,toilet]

    # Создаем словарь для хранения отфильтрованных контуров
    filtered_contours1 = {}

    # Обрабатываем все категории в одном цикле
    for category, contours in classes_dict.items():
        color1 = (randrange(50, 256), randrange(50, 256), randrange(50, 256))
        # Проверяем, что список контуров не пустой
        if contours and len(contours) > 0:
            # Инициализируем список для отфильтрованных контуров этой категории
            filtered_contours1[category] = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Проверяем границы
                if (x > border_size + border_margin
                        and y > border_size + border_margin
                        and (x + w) < (image_width - border_size - border_margin)
                        and (y + h) < (image_height - border_size - border_margin)):
                    # Добавляем в отфильтрованный список
                    filtered_contours1[category].append(contour)

                    # Рисуем контур (особая толщина для toilet)
                    thickness = 2
                    cv2.drawContours(bordered_image, [contour], 0, color1, thickness)
        else:
            # Если контуров нет, создаем пустой список для этой категории
            filtered_contours1[category] = []

    contours_wall = average_close2points(contours_wall, 4)
    contours_wall = average_close2points(contours_wall, 2)

    for i, contour in enumerate(contours_wall):
        contour = np.array( average_close_points(   contour.reshape(-1, 2), 3))
        print(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (x > border_size + border_margin
                and y > border_size + border_margin
                and (x + w) < (image_width - border_size - border_margin)
                and (y + h) < (image_height - border_size - border_margin)):
            filtered_contours_wall.append(contour)
            cv2.drawContours(bordered_image, [contour], 0, (randrange(50,256), randrange(50, 256), randrange(50, 256)), 1)
        # contour_points = contour.reshape(-1, 2)

    #     # Вывод таблицы для каждого контура
    #     print(f"\nКонтур #{i + 1}:")
    #     print(tabulate(
    #         contour_points,
    #         headers=['X', 'Y'],
    #         tablefmt="grid",
    #         showindex="always"
    #     ))
    # # print("________________________________________________________________________")
    # #
    # # print(f"\nКонтур #abs:")
    # # print(filtered_contours)
    # print("________________________________________________________________________")
    # print("________________________________________________________________________")

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
    # debug_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    #                     for img in debug_images]
    #
    # titles = [
    #     'Original Image',
    #     'Bordered Image',
    #     'Grayscale',
    #     'Binary Image',
    #     'Cleaned Image',
    #     'processed Image'
    # ]
    #
    # plt.figure(figsize=(15, 10))
    # for i, (img, title) in enumerate(zip(debug_images_rgb, titles)):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(img, cmap='gray' if i > 1 else None)
    #     plt.title(title)
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    # # _________________________________________________________________________________________
    # cv2.imwrite('output.png', cleaned_image)

    if not filtered_contours_wall:
        raise ValueError("Не обнаружено стен. Проверьте параметры обработки изображения.")

    # Корректировка координат контуров (удаление рамки)
    wall_contours = [cnt - border_size for cnt in filtered_contours_wall]

    # return sorted_corner, (original_height, original_width)
    return wall_contours, (original_height, original_width), filtered_contours1
