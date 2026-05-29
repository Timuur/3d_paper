import cv2
import numpy as np
from random import randrange

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
    import os
    import tempfile

    # 1. Загружаем изображение
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    original_height, original_width = original_image.shape[:2]
    print(f"\n📐 Исходный размер изображения: {original_width}×{original_height} px")

    # 2. Функция для поиска и сбора размеров дверей
    def extract_door_sizes(img_path):
        from check_img_ai_v2 import get_coord
        detections, labels = get_coord(img_path)
        bath_doors, regular_doors = [], []

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            door_size_px = max(xmax - xmin, ymax - ymin)

            # Фильтр по разумному размеру двери в пикселях
            if classname in ['door_bath', 'Door'] and 50 < door_size_px < 300:
                if classname == 'door_bath':
                    bath_doors.append(door_size_px)
                    print(f"   🚪 Дверь в ванную: {door_size_px} px")
                else:
                    regular_doors.append(door_size_px)
                    print(f"   🚪 Обычная дверь: {door_size_px} px")
        return bath_doors, regular_doors

    # 3. Первый проход: ищем двери на исходном изображении
    door_bath_pixels, door_regular_pixels = extract_door_sizes(image_path)
    current_cm_per_pixel = None

    # 4. Вычисляем масштаб по найденным дверям (приоритет: ванная → обычная)
    if door_bath_pixels:
        avg_px = np.median(door_bath_pixels)
        current_cm_per_pixel = target_bath_door_cm / avg_px
        print(f"\n📏 Масштаб по двери в ванную: {avg_px:.1f} px = {target_bath_door_cm} см")
        print(f"   Текущий масштаб: 1 px = {current_cm_per_pixel:.4f} см")
    elif door_regular_pixels:
        avg_px = np.median(door_regular_pixels)
        current_cm_per_pixel = target_regular_door_cm / avg_px
        print(f"\n📏 Масштаб по обычной двери: {avg_px:.1f} px = {target_regular_door_cm} см")
        print(f"   Текущий масштаб: 1 px = {current_cm_per_pixel:.4f} см")

    # 5. 🔧 Если двери не найдены — масштабируем план до 1500px по большей стороне и ищем снова
    if current_cm_per_pixel is None:
        print("⚠️ Двери не найдены! Пропорционально масштабируем план до 1500px по большей стороне...")
        max_side = max(original_width, original_height)
        if max_side > 0:
            # Промежуточное масштабирование для поиска дверей
            temp_scale = 1500.0 / max_side
            temp_width = int(original_width * temp_scale)
            temp_height = int(original_height * temp_scale)
            temp_image = cv2.resize(original_image, (temp_width, temp_height),
                                    interpolation=cv2.INTER_AREA if temp_scale < 1 else cv2.INTER_LANCZOS4)

            # Сохраняем во временный файл для анализа нейросетью
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
            cv2.imwrite(temp_path, temp_image)

            # Повторный поиск дверей
            door_bath_pixels, door_regular_pixels = extract_door_sizes(temp_path)

            # Очищаем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Пересчитываем масштаб, если двери найдены после ресайза
            if door_bath_pixels:
                avg_px = np.median(door_bath_pixels)
                # Масштаб относительно исходного изображения
                current_cm_per_pixel = (target_bath_door_cm / avg_px) * temp_scale
                print(f"✅ После ресайза найдена дверь в ванную: {avg_px:.1f} px")
            elif door_regular_pixels:
                avg_px = np.median(door_regular_pixels)
                current_cm_per_pixel = (target_regular_door_cm / avg_px) * temp_scale
                print(f"✅ После ресайза найдена обычная дверь: {avg_px:.1f} px")

    # 6. Фоллбэк: если дверей так и нет — используем дефолтный масштаб
    if current_cm_per_pixel is None:
        current_cm_per_pixel = 0.1  # 1px = 0.1 см по умолчанию
        print("⚠️ Двери так и не найдены, используем масштаб по умолчанию: 1px = 0.1 см")

    # 7. Вычисляем итоговый коэффициент масштабирования к целевому масштабу
    scale_factor = current_cm_per_pixel / target_cm_per_pixel
    print(f"\n🔄 Финальное масштабирование:")
    print(f"   Текущий масштаб: 1 px = {current_cm_per_pixel:.4f} см")
    print(f"   Целевой масштаб: 1 px = {target_cm_per_pixel:.4f} см")
    print(f"   Коэффициент: {scale_factor:.4f}")

    # 8. Вычисляем новые размеры
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    print(f"\n📐 Итоговые размеры: {original_width}×{original_height} → {new_width}×{new_height} px")

    # 9. Масштабируем изображение
    scaled_image = cv2.resize(
        original_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LANCZOS4
    )

    # 10. Сохраняем если указан путь
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
    # try:
    #     from check_img_ai_v2 import clear_cache as cache_clear_ii
    #     cache_clear_ii()
    # except:
    #     pass

    # # 3. Сброс окон OpenCV (если были открыты)
    # try:
    #     import cv2
    #     cv2.destroyAllWindows()
    # except:
    #     pass

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

def process_floor_plan(image_path, border_size=20, merge_objects=True, axis_tol=5.0, thick_tol=5.0, gap_tol=250.0):
    """
    Основная функция обработки плана помещения
    :param image_path: Путь к исходному изображению
    :param border_size: Размер обрамляющей рамки
    :param merge_objects: Если True — объединять коллинеарные прямоугольники (по умолчанию True)
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
        xyy = xy1, xy2, xy3, xy4

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        if classname in classes_dict:
            classes_dict[classname].append(np.array(xyy))

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

        else:
            # Если контуров нет, создаем пустой список для этой категории
            filtered_contours1[category] = []

    contours_wall = average_close2points(contours_wall, 4)
    contours_wall = average_close2points(contours_wall, 2)

    for i, contour in enumerate(contours_wall):
        print(contour)
        contour = np.array(average_close_points(contour.reshape(-1, 2), 3))
        x, y, w, h = cv2.boundingRect(contour)
        if (x > border_size + border_margin
                and y > border_size + border_margin
                and (x + w) < (image_width - border_size - border_margin)
                and (y + h) < (image_height - border_size - border_margin)):
            filtered_contours_wall.append(contour)

    if merge_objects:
        print("\n🔄 Запуск пайплайна объединения объектов...")
        filtered_contours1 = merge_rects_pipeline(
            filtered_contours1,
            axis_tol=axis_tol,# Допуск смещения оси (пиксели)
            thick_tol=thick_tol,# Допуск разницы толщины
            gap_tol=gap_tol,# Максимальный разрыв для объединения
            verbose=True
        )
        print("✅ Объединение завершено")
        print("\n🔄 Запуск пайплайна объединения СТЕН...")
        filtered_contours_wall = merge_wall_contours(
            filtered_contours_wall,
            axis_tol=axis_tol,
            thick_tol=thick_tol,
            gap_tol=gap_tol,
            verbose=True
        )
        print("✅ Объединение стен завершено")

    print(f"\nКонтур #wall:")
    print(filtered_contours_wall)
    print("________________________________________________________________________")

    print(f"\nКонтур #abs:")
    print(filtered_contours1)
    print("________________________________________________________________________")

    if not filtered_contours_wall:
        raise ValueError("Не обнаружено стен. Проверьте параметры обработки изображения.")

    # Корректировка координат контуров (удаление рамки)
    wall_contours = [cnt - border_size for cnt in filtered_contours_wall]

    for key in filtered_contours1:
        filtered_contours1[key] = [cnt - border_size for cnt in filtered_contours1[key]]

    # return sorted_corner, (original_height, original_width)
    return wall_contours, (original_height, original_width), filtered_contours1

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from collections import defaultdict


def decompose_orthogonal_polygon(coords):
    """
    1. РАЗБИВКА: Преобразует сложный ортогональный полигон в набор простых прямоугольников.
    Использует метод сетки по уникальным координатам вершин.
    """
    # Конвертируем в numpy array если нужно
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    # Убираем лишние измерения если есть
    if coords.ndim == 3 and coords.shape[1] == 1:
        coords = coords.squeeze(1)

    poly = Polygon(coords)
    if not poly.is_valid or poly.is_empty:
        return []

    xs = sorted(set(coords[:, 0]))
    ys = sorted(set(coords[:, 1]))

    rects = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx = (xs[i] + xs[i + 1]) / 2
            cy = (ys[j] + ys[j + 1]) / 2
            # Если центр ячейки внутри исходного полигона -> сохраняем прямоугольник
            if poly.contains(Point(cx, cy)):
                rects.append(np.array([
                    [xs[i], ys[j]], [xs[i + 1], ys[j]],
                    [xs[i + 1], ys[j + 1]], [xs[i], ys[j + 1]]
                ]))
    return rects


def merge_rects_pipeline(plan_data, axis_tol=5.0, thick_tol=5.0, gap_tol=250.0, verbose=False):
    """
    Полный пайплайн: Декомпозиция -> Группировка -> Объединение -> Возврат координат
    Возвращает данные в том же формате: dict[str, list[np.array]]
    """
    final_plan = {}
    stats = {}
    gap_tol = max(gap_tol, 0.0)

    target_cat = {'Wall', 'h-wall'}

    for category, polygons in plan_data.items():
        if category not in target_cat:
            final_plan[category] = polygons  # Возвращаем без изменений
            continue
        if not polygons:
            final_plan[category] = []
            continue

        # 1️⃣ ИЗВЛЕЧЕНИЕ СВОЙСТВ (без декомпозиции — работаем напрямую с прямоугольниками)
        props = []
        for rect_coords in polygons:
            # Нормализация входных данных: приводим к numpy array (N, 2)
            if isinstance(rect_coords, (list, tuple)):
                coords = np.array(rect_coords)
            elif isinstance(rect_coords, np.ndarray):
                coords = rect_coords.squeeze() if rect_coords.ndim == 3 else rect_coords
            else:
                continue

            if coords.shape != (4, 2):
                # Если это не 4 точки — пропускаем (или можно добавить логирование)
                continue

            p = Polygon(coords)
            if not p.is_valid or p.area == 0:
                continue

            minx, miny, maxx, maxy = p.bounds
            w, h = maxx - minx, maxy - miny

            if w >= h:  # Горизонтальный прямоугольник
                props.append({'poly': p, 'orient': 'H', 'thick': h,
                              'axis': (miny + maxy) / 2, 'start': minx, 'end': maxx})
            else:  # Вертикальный прямоугольник
                props.append({'poly': p, 'orient': 'V', 'thick': w,
                              'axis': (minx + maxx) / 2, 'start': miny, 'end': maxy})

        if not props:
            final_plan[category] = []
            continue

        # 2️⃣ ГРУППИРОВКА (по ориентации, толщине, оси с учётом допусков)
        groups = defaultdict(list)
        for p in props:
            t_key = p['thick'] if thick_tol == 0 else round(p['thick'] / thick_tol) * thick_tol
            a_key = p['axis'] if axis_tol == 0 else round(p['axis'] / axis_tol) * axis_tol
            key = (p['orient'], t_key, a_key)
            groups[key].append(p)

        # 4️⃣ ОБЪЕДИНЕНИЕ (линейный проход с проверкой зазора)
        merged_rects = []
        merged_count = 0
        for key, items in groups.items():
            items.sort(key=lambda x: x['start'])
            if not items: continue

            curr = items[0]
            for nxt in items[1:]:
                # Проверка разрыва <= gap_tol
                if nxt['start'] <= curr['end'] + gap_tol:
                    curr['end'] = max(curr['end'], nxt['end'])
                    curr['poly'] = unary_union([curr['poly'], nxt['poly']])
                    merged_count += 1
                else:
                    mb = curr['poly'].bounds
                    merged_rects.append(np.array([
                        [mb[0], mb[1]], [mb[2], mb[1]],
                        [mb[2], mb[3]], [mb[0], mb[3]]
                    ]))
                    curr = nxt

            mb = curr['poly'].bounds
            merged_rects.append(np.array([
                [mb[0], mb[1]], [mb[2], mb[1]],
                [mb[2], mb[3]], [mb[0], mb[3]]
            ]))

        final_plan[category] = merged_rects
        stats[category] = {
            'before': len(polygons),
            'after': len(merged_rects),
            'unions_performed': merged_count
        }

    if verbose and stats:
        print("\n📊 СТАТИСТИКА ОБЪЕДИНЕНИЯ (только Wall / h-wall)")
        print(f"{'Категория':<15} | {'Исходных':<8} | {'Итого':<8} | Объед.")
        print("-" * 50)
        for cat, s in stats.items():
            if s['before'] > 0:
                print(f"{cat:<15} | {s['before']:<8} | {s['after']:<8} | {s['unions_performed']}")
        if not stats:
            print(f"⚠️ Целевые категории {target_cat} не найдены или пусты")

    return final_plan


def merge_wall_contours(wall_contours, axis_tol=5.0, thick_tol=5.0, gap_tol=250.0, verbose=False):
    """
    Обрабатывает список сложных контуров стен:
    1. Декомпозиция на простые прямоугольники
    2. Группировка по ориентации/толщине/оси
    3. Объединение коллинеарных сегментов с учётом зазора
    4. Возврат в формате List[np.array] + опциональная таблица статистики
    """
    if not wall_contours:
        return []

    gap_tol = max(gap_tol, 0.0)
    n_walls = len(wall_contours)

    # 📊 Инициализация статистики по каждой стене
    wall_stats = [{'decomposed': 0, 'merges': 0, 'final_indices': set()} for _ in range(n_walls)]

    # 1️⃣ РАЗБИВКА всех сложных контуров на простые прямоугольники
    decomposed = []
    for i, contour in enumerate(wall_contours):
        if isinstance(contour, np.ndarray):
            coords = contour.squeeze() if contour.ndim == 3 else contour
        else:
            coords = np.array(contour)

        rects = decompose_orthogonal_polygon(coords)
        for r in rects:
            # Сохраняем индекс исходной стены
            decomposed.append((r, i))

    if not decomposed:
        return wall_contours  # Возвращаем как есть, если не удалось разбить

    # 2️⃣ ИЗВЛЕЧЕНИЕ СВОЙСТВ
    props = []
    for coords, src_idx in decomposed:
        p = Polygon(coords)
        if not p.is_valid: continue

        # Считаем, на сколько частей разбилась стена
        wall_stats[src_idx]['decomposed'] += 1

        minx, miny, maxx, maxy = p.bounds
        w, h = maxx - minx, maxy - miny

        if w >= h:  # Горизонтальный
            props.append({'poly': p, 'source': src_idx, 'orient': 'H', 'thick': h,
                          'axis': (miny + maxy) / 2, 'start': minx, 'end': maxx})
        else:  # Вертикальный
            props.append({'poly': p, 'source': src_idx, 'orient': 'V', 'thick': w,
                          'axis': (minx + maxx) / 2, 'start': miny, 'end': maxy})

    # 3️⃣ ГРУППИРОВКА
    groups = defaultdict(list)
    for p in props:
        t_key = p['thick'] if thick_tol == 0 else round(p['thick'] / thick_tol) * thick_tol
        a_key = p['axis'] if axis_tol == 0 else round(p['axis'] / axis_tol) * axis_tol
        key = (p['orient'], t_key, a_key)
        groups[key].append(p)

    # 4️⃣ ОБЪЕДИНЕНИЕ
    merged_rects = []
    total_merged_count = 0
    final_seg_idx = 0

    for key, items in groups.items():
        items.sort(key=lambda x: x['start'])
        if not items: continue

        # Добавляем tracking источников для корректного учёта при слиянии
        for item in items:
            item['src_set'] = {item['source']}

        curr = items[0]
        for nxt in items[1:]:
            if nxt['start'] <= curr['end'] + gap_tol:
                curr['end'] = max(curr['end'], nxt['end'])
                curr['poly'] = unary_union([curr['poly'], nxt['poly']])

                # 🔢 Учёт слияний: инкрементируем для всех стен, чьи куски участвуют
                for src in curr['src_set']:
                    wall_stats[src]['merges'] += 1
                for src in nxt['src_set']:
                    if src not in curr['src_set']:
                        wall_stats[src]['merges'] += 1

                curr['src_set'].update(nxt['src_set'])
                total_merged_count += 1
            else:
                # 🏁 Фиксируем итоговый сегмент
                for src in curr['src_set']:
                    wall_stats[src]['final_indices'].add(final_seg_idx)
                final_seg_idx += 1

                mb = curr['poly'].bounds
                merged_rects.append(np.array([
                    [mb[0], mb[1]], [mb[2], mb[1]],
                    [mb[2], mb[3]], [mb[0], mb[3]]
                ]))
                curr = nxt

        # Фиксируем последний сегмент в группе
        for src in curr['src_set']:
            wall_stats[src]['final_indices'].add(final_seg_idx)
        final_seg_idx += 1

        mb = curr['poly'].bounds
        merged_rects.append(np.array([
            [mb[0], mb[1]], [mb[2], mb[1]],
            [mb[2], mb[3]], [mb[0], mb[3]]
        ]))

    # 📊 Вывод таблицы при verbose=True
    if verbose:
        print("\n📊 Статистика обработки стен:")
        print(f"{'Стена':<6} | {'Разбилось на':<12} | {'Слияний':<9} | {'Итоговых сегментов':<19}")
        print("-" * 55)
        for i, stat in enumerate(wall_stats):
            print(f"{i:<6} | {stat['decomposed']:<12} | {stat['merges']:<9} | {len(stat['final_indices']):<19}")
        print(
            f"\n🧱 Итого: {n_walls} сложных → {len(decomposed)} простых → {len(merged_rects)} объединённых (+{total_merged_count} слияний)\n")

    return merged_rects