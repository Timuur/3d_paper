import cv2
import numpy as np
from random import randrange

from matplotlib import pyplot as plt

from check_img_ai_v2 import get_coord

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
                'h-wall': [],
                'door_vhod_l': [],
                'sink': [],
                'sink_kitchen': [],
                'balcon_wall': [],
                'toulet': [],
                'wash_machine': [],
                'win_in_wall': []
                }


def calculate_and_resize_image(image_path, output_path=None, target_bath_door_cm=90, target_regular_door_cm=100, target_cm_per_pixel=0.5):
    """
    Масштабирует изображение плана так, чтобы двери соответствовали стандартным размерам.

    Args:
        image_path: Путь к исходному изображению
        output_path: Путь для сохранения масштабированного изображения (если None - не сохранять)
        target_bath_door_cm: Целевая ширина двери в ванную (см)
        target_regular_door_cm: Целевая ширина обычной двери (см)

    Returns:
        dict: {
            'scaled_image': масштабированное изображение,
            'scale_factor': коэффициент масштабирования,
            'original_size': (width, height) исходного изображения,
            'new_size': (width, height) нового изображения,
            'cm_per_pixel': масштаб (см/пиксель) после масштабирования
        }
    """
    # 1. Загружаем изображение и детектируем объекты
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    original_height, original_width = original_image.shape[:2]
    print(f"\n📐 Исходный размер изображения: {original_width}×{original_height} px")

    # 2. Получаем координаты дверей (предполагаем, что get_coord уже настроен)
    from check_img_ai_v2 import get_coord

    # Временная обработка для получения дверей
    detections, labels = get_coord(image_path)

    door_bath_pixels = []
    door_regular_pixels = []

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        width_px = xmax - xmin
        height_px = ymax - ymin
        door_size_px = max(width_px, height_px)  # Большая сторона - это ширина двери

        # Классифицируем двери
        if classname in ['door_bath', 'Door'] and 50 < door_size_px < 300:
            if classname == 'door_bath':
                door_bath_pixels.append(door_size_px)
                print(f"   🚪 Дверь в ванную: {door_size_px} px")
            else:
                door_regular_pixels.append(door_size_px)
                print(f"   🚪 Обычная дверь: {door_size_px} px")

    # 3. Вычисляем целевой масштаб
    if door_bath_pixels:
        # Приоритет: дверь в ванную
        avg_bath_door_px = np.median(door_bath_pixels)
        current_cm_per_pixel = target_bath_door_cm / avg_bath_door_px

        print(f"\n📏 Масштаб определен по двери в ванную:")
        print(f"   Средняя ширина: {avg_bath_door_px:.1f} px")
        print(f"   Целевой размер: {target_bath_door_cm} см")
        print(f"   Текущий масштаб: 1 px = {current_cm_per_pixel:.4f} см")

    elif door_regular_pixels:
        # Если нет двери в ванную - используем обычные двери
        avg_regular_door_px = np.median(door_regular_pixels)
        current_cm_per_pixel = target_regular_door_cm / avg_regular_door_px

        print(f"\n📏 Масштаб определен по обычной двери:")
        print(f"   Средняя ширина: {avg_regular_door_px:.1f} px")
        print(f"   Целевой размер: {target_regular_door_cm} см")
        print(f"   Текущий масштаб: 1 px = {current_cm_per_pixel:.4f} см")
    else:
        print("⚠️ Двери не найдены! Используется масштаб по умолчанию")
        current_cm_per_pixel = 0.1  # Дефолтный масштаб

    # 4. Вычисляем желаемый масштаб
    # Целевой масштаб: например, 1 пиксель = 0.5 см (или другой удобный)
    # target_cm_per_pixel = 0.5  # Можно настроить

    # Коэффициент масштабирования изображения
    scale_factor = current_cm_per_pixel / target_cm_per_pixel

    print(f"\n🔄 Масштабирование:")
    print(f"   Текущий масштаб: 1 px = {current_cm_per_pixel:.4f} см")
    print(f"   Целевой масштаб: 1 px = {target_cm_per_pixel:.4f} см")
    print(f"   Коэффициент: {scale_factor:.4f}")

    # 5. Вычисляем новые размеры
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    print(f"\n📐 Новые размеры:")
    print(f"   {original_width}×{original_height} → {new_width}×{new_height} px")

    # 6. Масштабируем изображение
    scaled_image = cv2.resize(
        original_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
    )

    # 7. Сохраняем если указан путь
    if output_path:
        cv2.imwrite(output_path, scaled_image)
        print(f"✅ Изображение сохранено: {output_path}")

    return {
        'scaled_image': scaled_image,
        'scale_factor': scale_factor,
        'original_size': (original_width, original_height),
        'new_size': (new_width, new_height),
        'cm_per_pixel': target_cm_per_pixel
    }

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
            else:  # Если массив вида [x, y]
                x1, y1 = points[i][j].ravel()

            # Сравниваем со всеми остальными точками
            for k in range(len(points)):
                for l in range(len(points[k])):
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

def clear_cache():
    """Полный сброс состояния модуля img2wall"""
    global classes_dict

    # 1. Очистка основного источника "кэша"
    for key in classes_dict:
        classes_dict[key] = []

    # 2. Безопасная очистка кэша нейросети
    try:
        from check_img_ai_v2 import clear_cache as cache_clear_ii
        cache_clear_ii()
    except:
        pass

    # 3. Сброс окон OpenCV (если были открыты)
    try:
        import cv2
        cv2.destroyAllWindows()
    except:
        pass

def replace_gray_in_monochrome(image, lower_gray=39, upper_gray=255, target_value=255):
    """
    Заменяет диапазон серых пикселей в монохромном изображении на белые.

    Параметры:
    - image: одноканальное изображение (numpy array).
    - lower_gray: нижняя граница серого (0-255).
    - upper_gray: верхняя граница серого (0-255).
    - target_value: целевое значение (по умолчанию 255 — белый).
    """
    mask = (image >= lower_gray) & (image <= upper_gray)
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
            # print('(', scale_p,")hight_d = ", pixel_on_map, ' / ', wight_d_i,'/', wight_d_i*0.05)
        else:
            # scale_win = hight_d_i * float(scale) / hight_win[0]
            scale_p =  pixel_on_map / hight_d_i
            # print('(', scale_p,")hight_d = ", pixel_on_map, '/', hight_d_i,'/', hight_d_i*0.05)
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

    global classes_dict
    for key in classes_dict:
        classes_dict[key] = []
    #TODO:Фабрика свежего словаря классов на основе классов ии

    # Загрузка изображения и предобработка
    original_image = cv2.imread(image_path)


    # _________________________________________________________________________________________
    # _________________________________________________________________________________________

    # class_path_txt = get_file_path('ai_model/classes.txt')


    detections, labels = get_coord(image_path)
    # print(detections)
    print(labels)
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
                    # thickness = 2
                    # cv2.drawContours(bordered_image, [contour], 0, color1, thickness)
        else:
            # Если контуров нет, создаем пустой список для этой категории
            filtered_contours1[category] = []

    contours_wall = average_close2points(contours_wall, 4)
    contours_wall = average_close2points(contours_wall, 2)

    for i, contour in enumerate(contours_wall):
        print(contour)
        contour = np.array( average_close_points(   contour.reshape(-1, 2), 3))
        x, y, w, h = cv2.boundingRect(contour)
        if (x > border_size + border_margin
                and y > border_size + border_margin
                and (x + w) < (image_width - border_size - border_margin)
                and (y + h) < (image_height - border_size - border_margin)):
            filtered_contours_wall.append(contour)
            # cv2.drawContours(bordered_image, [contour], 0, (randrange(50,256), randrange(50, 256), randrange(50, 256)), 1)
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
    print(f"\nКонтур #abs:")
    print(filtered_contours1)
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

    for key in filtered_contours1:
        filtered_contours1[key] = [cnt - border_size for cnt in filtered_contours1[key]]

    # return sorted_corner, (original_height, original_width)
    return wall_contours, (original_height, original_width), filtered_contours1


def process_floor_plan_with_scaling(image_path, border_size=20, target_cm_per_pixel=0.5):
    """
    Обработка плана с автоматическим масштабированием изображения.

    Args:
        image_path: Путь к изображению
        border_size: Размер рамки
        target_cm_per_pixel: Целевой масштаб (см/пиксель)

    Returns:
        wall_contours, original_size, obj_contours, cm_per_pixel
        (БЕЗ scaled_image - оно уже возвращено через calculate_and_resize_image)
    """
    global classes_dict

    # 1. Сначала масштабируем изображение
    print("=" * 60)
    print("📐 ЭТАП 1: МАСШТАБИРОВАНИЕ ИЗОБРАЖЕНИЯ")
    print("=" * 60)

    resize_result = calculate_and_resize_image(
        image_path,
        output_path=None,  # Можно указать 'scaled_plan.png'
        target_bath_door_cm=90,
        target_regular_door_cm=100
    )

    scaled_image = resize_result['scaled_image']
    cm_per_pixel = resize_result['cm_per_pixel']

    # 2. Временное сохранение для обработки
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, scaled_image)

    try:
        # 3. Обрабатываем масштабированное изображение
        print("\n" + "=" * 60)
        print("📐 ЭТАП 2: ОБРАБОТКА МАСШТАБИРОВАННОГО ПЛАНА")
        print("=" * 60)

        # Очищаем кэш
        classes_dict = {key: [] for key in classes_dict}

        # Детектируем объекты на масштабированном изображении
        from check_img_ai_v2 import get_coord

        detections, labels = get_coord(temp_path)

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Добавляем border_size
            xy1 = (xmin + border_size, ymin + border_size)
            xy2 = (xmin + border_size, ymax + border_size)
            xy3 = (xmax + border_size, ymax + border_size)
            xy4 = (xmax + border_size, ymin + border_size)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            if classname in classes_dict:
                contour = np.array([xy1, xy2, xy3, xy4])
                classes_dict[classname].append(contour)

        # 4. Обрабатываем стены (как раньше)
        original_height, original_width = scaled_image.shape[:2]
        bordered_image = add_white_border(scaled_image, border_size)

        gray_image = cv2.cvtColor(bordered_image, cv2.COLOR_BGR2GRAY)
        processed_image = apply_adaptive_threshold(gray_image)

        # Морфология и очистка
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Поиск контуров стен
        contours_wall, _ = cv2.findContours(cleaned_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация
        filtered_contours_wall = []
        border_margin = 10
        image_height, image_width = cleaned_image.shape

        for contour in contours_wall:
            contour = contour.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(contour)

            if (x > border_size + border_margin and
                    y > border_size + border_margin and
                    (x + w) < (image_width - border_size - border_margin) and
                    (y + h) < (image_height - border_size - border_margin)):
                filtered_contours_wall.append(contour)

        # Усреднение точек
        filtered_contours_wall = average_close2points(filtered_contours_wall, 4)

        # Корректировка координат (убираем border_size)
        wall_contours = [cnt - border_size for cnt in filtered_contours_wall]

        # Корректируем координаты объектов
        for key in classes_dict:
            classes_dict[key] = [cnt - border_size for cnt in classes_dict[key]]

        print(f"\n✅ Обработка завершена!")
        print(f"   Найдено стен: {len(wall_contours)}")
        print(f"   Найдено объектов: {sum(len(v) for v in classes_dict.values())}")
        print(f"   Масштаб: 1 px = {cm_per_pixel:.4f} см")

        # ВОЗВРАЩАЕМ 4 ЗНАЧЕНИЯ (без scaled_image)
        return wall_contours, (original_height, original_width), classes_dict, cm_per_pixel

    finally:
        # Удаляем временный файл
        import os
        try:
            os.unlink(temp_path)
        except:
            pass