import sys
from pathlib import Path
import numpy as np
import trimesh
import json

import img2wall as i2w
from scipy.spatial import ConvexHull
from trimesh.transformations import rotation_matrix, concatenate_matrices, translation_matrix
from shapely.geometry import Polygon

import logging

from check_ddor import extract_wall_segments_with_ids, find_true_opening_center


# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
# logger = logging.getLogger(__name__)

def get_file_path(filename: str) -> Path:
    base_path = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
    return base_path / filename


def reload_mesh_paths() -> dict:
    """
    Принудительная перезагрузка конфигурации путей.
    Полезно при изменении JSON во время работы приложения.
    """
    global obj_mesh
    obj_mesh = load_mesh_paths()
    return obj_mesh


def clear_cache():
    """Очистка кэшей генератора 3D"""
    import gc

    try:
        if hasattr(trimesh, 'cache'):
            trimesh.cache.clear()
    except:
        pass

    # Также можно очистить кэш загрузчиков
    if hasattr(trimesh.exchange, 'load') and hasattr(trimesh.exchange.load, '_mesh_loaders'):
        pass

    reload_mesh_paths()
    gc.collect()


def load_mesh_paths() -> dict:
    """
    Загружает пути к 3D-моделям из JSON-конфига.
    Возвращает словарь {class_name: relative_path}.
    """
    config_path = get_file_path('config/mesh_paths.json')

    if not config_path.exists():
        print(f"⚠️  Файл конфигурации не найден: {config_path}")
        print("📝  Используются пути по умолчанию (пустые)")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            mesh_paths = json.load(f)

        # Валидация: проверяем, что все значения — строки
        for key, value in mesh_paths.items():
            if not isinstance(value, str):
                print(f"⚠️  Неверный тип пути для '{key}': {type(value)}. Ожидалась строка.")
                mesh_paths[key] = ""

        print(f"✅  Загружено {len(mesh_paths)} путей к 3D-моделям из {config_path.name}")
        return mesh_paths

    except json.JSONDecodeError as e:
        print(f"❌  Ошибка парсинга JSON в {config_path}: {e}")
        print("📝  Используйте валидный JSON (проверьте запятые и кавычки)")
        return {}
    except Exception as e:
        print(f"❌  Ошибка загрузки конфигурации: {e}")
        return {}


obj_mesh = load_mesh_paths()


def _load_mesh_safe(path: Path, default_scale: float = 1.0):
    """Безопасная загрузка меша с валидацией."""
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh.is_empty:
            print(f"❌ Пустой меш: {path}")
            return None
        mesh.apply_scale(default_scale)
        mesh.fix_normals()
        print(f"✅ Загружен меш: {path.name}, вершин: {len(mesh.vertices)}")
        print(f"   Базовые размеры: {mesh.extents}")
        return mesh
    except Exception as e:
        print(f"❌ Ошибка загрузки {path}: {e}")
        return None


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


def _determine_object_orientation(contour):
    """
    Определяет ориентацию объекта по его bounding box.

    Returns:
        tuple: (is_vertical, width_px, height_px, angle_rad)
        - is_vertical: True если высота > ширины
        - width_px: ширина в пикселях
        - height_px: высота в пикселях
        - angle_rad: угол поворота вокруг Z (для вертикальных объектов)
    """
    width_px = abs(contour[2][0] - contour[0][0])
    height_px = abs(contour[1][1] - contour[0][1])

    is_vertical = height_px > width_px

    # Для вертикальных объектов нужен поворот на 90 градусов вокруг Z
    angle_rad = np.pi / 2 if is_vertical else 0

    return is_vertical, width_px, height_px, angle_rad


def build_obj(obj_contours, wall_contours, scale_cm_per_pixel, height_cm=270):
    """
    Строит 3D модели объектов (мебель, сантехника и т.д.)

    Args:
        scale_cm_per_pixel: масштаб (см/пиксель)
        height_cm: высота помещения (для позиционирования)
    """
    scene_objects = []

    for class_o in obj_contours:
        if class_o in ["Door", "Window"]:
            continue

        if class_o not in obj_mesh or not obj_mesh[class_o]:
            continue

        print(f"\n🪑 Обработка объектов класса: {class_o}")
        mesh_obj = _load_mesh_safe(get_file_path(obj_mesh[class_o]), default_scale=1.0)

        if mesh_obj is None:
            continue

        # Базовые размеры модели
        base_extents = mesh_obj.extents.copy()
        print(f"   Базовые размеры модели: {base_extents}")

        for i, contour in enumerate(obj_contours[class_o]):
            try:
                # 1. Определяем ориентацию
                is_vertical, width_px, height_px, angle_z = _determine_object_orientation(contour)

                # 2. Вычисляем размеры в см
                width_cm = width_px * scale_cm_per_pixel
                depth_cm = height_px * scale_cm_per_pixel

                # 3. Центр объекта (в пикселях → см)
                center_px = contour.mean(axis=0)
                center_cm = center_px * scale_cm_per_pixel

                # 4. Создаем копию меша
                mesh = mesh_obj.copy()

                # 5. Масштабируем к реальным размерам
                # Предполагаем: X - ширина, Y - глубина, Z - высота
                scale_x = width_cm / base_extents[0] if base_extents[0] > 0 else 1.0
                scale_y = depth_cm / base_extents[1] if base_extents[1] > 0 else 1.0
                # Высота остается как есть или масштабируется пропорционально

                mesh.apply_scale([scale_x, scale_y, 1.0])

                # 6. Строим трансформацию
                transform = np.eye(4)

                # Поворот вокруг Z для вертикальных объектов
                if angle_z > 0:
                    rot_z = rotation_matrix(angle_z, [0, 0, 1])
                    transform = concatenate_matrices(transform, rot_z)

                # Поворот из XY плоскости (план) в XZ плоскость (3D)
                # Объект на плане лежит в XY, в 3D должен стоять на полу (XZ)
                rot_x = rotation_matrix(-np.pi / 2, [1, 0, 0])
                transform = concatenate_matrices(transform, rot_x)

                # 7. Позиция: центр объекта, на уровне пола
                # После поворота вокруг X на -90°, Y становится Z
                obj_height = mesh.extents[2]
                position_z = obj_height / 2  # Поднимаем на половину высоты

                transform[:3, 3] = [center_cm[0], -center_cm[1], position_z]

                mesh.apply_transform(transform)

                scene_objects.append(mesh)

                print(f"   ✅ {class_o} #{i + 1}: {width_cm:.1f}×{depth_cm:.1f} см, "
                      f"ориентация: {'вертикальная' if is_vertical else 'горизонтальная'}")

            except Exception as e:
                print(f"   ❌ Ошибка {class_o} #{i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return scene_objects


def build_door(door_contours, wall_contours, scale_cm_per_pixel, height_cm=270):
    """
    Строит 3D модели дверей

    Args:
        scale_cm_per_pixel: масштаб (см/пиксель)
        height_cm: высота помещения (см)
    """
    scene_objects = []
    segments = extract_wall_segments_with_ids(wall_contours)

    mesh_door = _load_mesh_safe(get_file_path(obj_mesh['Door']), default_scale=1.0)
    if mesh_door is None:
        print("❌ Не загружена модель двери")
        return []

    print(f"\n🚪 Построение дверей...")
    base_extents = mesh_door.extents.copy()
    print(f"   Базовые размеры модели: {base_extents}")

    for i, contour in enumerate(door_contours):
        try:
            # 1. Определяем ориентацию
            is_vertical, width_px, thickness_px, _ = _determine_object_orientation(contour)

            # 2. Размеры в см
            width_cm = width_px * scale_cm_per_pixel
            thickness_cm = thickness_px * scale_cm_per_pixel
            door_height_cm = 200  # Стандартная высота двери 200 см

            # 3. Центр двери (в пикселях → см)
            center_px = contour.mean(axis=0)
            center_cm = center_px * scale_cm_per_pixel

            # 4. Копия меша
            mesh = mesh_door.copy()

            # 5. Масштабирование
            scale_x = width_cm / base_extents[0] if base_extents[0] > 0 else 1.0
            scale_y = thickness_cm / base_extents[1] if base_extents[1] > 0 else 1.0
            scale_z = door_height_cm / base_extents[2] if base_extents[2] > 0 else 1.0

            mesh.apply_scale([scale_x, scale_y, scale_z])

            # 6. Трансформация
            transform = np.eye(4)

            # Для вертикальных дверей (когда thickness > width на плане)
            # нужен дополнительный поворот
            if is_vertical:
                rot_z = rotation_matrix(np.pi / 2, [0, 0, 1])
                transform = concatenate_matrices(transform, rot_z)

            # Поворот из XY (план) в XZ (3D, дверь стоит вертикально)
            rot_x = rotation_matrix(-np.pi / 2, [1, 0, 0])
            transform = concatenate_matrices(transform, rot_x)

            # 7. Позиция: дверь стоит на полу
            position_z = door_height_cm / 2  # Центр двери по высоте

            transform[:3, 3] = [center_cm[0], -center_cm[1], position_z]

            mesh.apply_transform(transform)

            # 8. Заполнитель стены над дверью (если нужно)
            # Для простоты пока не добавляем

            scene_objects.append(mesh)

            print(f"   ✅ Дверь #{i + 1}: {width_cm:.1f}×{thickness_cm:.1f}×{door_height_cm} см, "
                  f"ориентация: {'вертикальная' if is_vertical else 'горизонтальная'}")

        except Exception as e:
            print(f"   ❌ Ошибка двери #{i + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return scene_objects


def build_window(window_position, wall_contours, scale_cm_per_pixel, height_cm=270):
    """
    Строит 3D модели окон

    Args:
        scale_cm_per_pixel: масштаб (см/пиксель)
        height_cm: высота помещения (см)
    """
    scene_objects = []

    mesh_window = _load_mesh_safe(get_file_path(obj_mesh['Window']), default_scale=1.0)
    if mesh_window is None:
        print("❌ Не загружена модель окна")
        return []

    print(f"\n🪟 Построение окон...")
    base_extents = mesh_window.extents.copy()
    print(f"   Базовые размеры модели: {base_extents}")

    window_sill_height_cm = 90  # Высота подоконника от пола
    window_height_cm = 150  # Стандартная высота окна

    for i, contour in enumerate(window_position):
        try:
            # 1. Определяем ориентацию
            is_vertical, width_px, thickness_px, _ = _determine_object_orientation(contour)

            # 2. Размеры в см
            width_cm = width_px * scale_cm_per_pixel
            thickness_cm = thickness_px * scale_cm_per_pixel

            # 3. Центр окна
            center_px = contour.mean(axis=0)
            center_cm = center_px * scale_cm_per_pixel

            # 4. Копия меша
            mesh = mesh_window.copy()

            # 5. Масштабирование
            scale_x = width_cm / base_extents[0] if base_extents[0] > 0 else 1.0
            scale_y = thickness_cm / base_extents[1] if base_extents[1] > 0 else 1.0
            scale_z = window_height_cm / base_extents[2] if base_extents[2] > 0 else 1.0

            mesh.apply_scale([scale_x, scale_y, scale_z])

            # 6. Трансформация
            transform = np.eye(4)

            # Для вертикальных окон
            if is_vertical:
                rot_z = rotation_matrix(np.pi / 2, [0, 0, 1])
                transform = concatenate_matrices(transform, rot_z)

            # Поворот из XY в XZ
            rot_x = rotation_matrix(-np.pi / 2, [1, 0, 0])
            transform = concatenate_matrices(transform, rot_x)

            # Дополнительный поворот для правильной ориентации окна в стене
            rot_y = rotation_matrix(np.pi / 2, [0, 1, 0])
            transform = concatenate_matrices(transform, rot_y)

            # 7. Позиция: на высоте подоконника
            position_z = window_sill_height_cm + (window_height_cm / 2)

            transform[:3, 3] = [center_cm[0], -center_cm[1], position_z]

            mesh.apply_transform(transform)

            scene_objects.append(mesh)

            print(f"   ✅ Окно #{i + 1}: {width_cm:.1f}×{thickness_cm:.1f}×{window_height_cm} см, "
                  f"подоконник: {window_sill_height_cm} см")

        except Exception as e:
            print(f"   ❌ Ошибка окна #{i + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return scene_objects


def build_3d_model(wall_contours, scale_cm_per_pixel, height_cm=270):
    """
    Строит 3D-модель помещения экструдированием контуров стен

    Args:
        wall_contours: Контуры стен из обработки изображения
        scale_cm_per_pixel: Масштаб преобразования (см/пиксель)
        height_cm: Высота помещения в см (по умолчанию 270 см = 2.7 м)

    Returns:
        trimesh.Scene: Сцена с 3D-моделью
    """
    scene_objects = []
    all_cont = []

    print(f"\n🏗️ Построение стен...")
    print(f"   Масштаб: 1 px = {scale_cm_per_pixel:.4f} см")
    print(f"   Высота стен: {height_cm} см")

    for i, contour in enumerate(wall_contours):
        try:
            # Преобразуем контур в массив точек
            contour_points = np.array(contour).reshape(-1, 2)

            # Масштабируем координаты: пиксели → см
            scaled_cm = contour_points * scale_cm_per_pixel

            # Замыкаем контур при необходимости
            if not np.allclose(scaled_cm[0], scaled_cm[-1]):
                scaled_cm = np.vstack([scaled_cm, scaled_cm[0]])

            all_cont.append(scaled_cm)

            # Создаём 2D полигон (координаты в см)
            polygon = trimesh.path.polygons.Polygon(scaled_cm)

            # Экструдируем в 3D (высота в см)
            mesh = trimesh.creation.extrude_polygon(polygon, height=height_cm)

            # Инвертируем Y и Z для правильной ориентации
            # Image coords: Y вниз → World coords: Y вверх
            # Z вверх
            matrix_transform = np.array([
                [1, 0, 0, 0],  # X остается
                [0, -1, 0, 0],  # Y инвертируется
                [0, 0, -1, 0],  # Z инвертируется (вверх)
                [0, 0, 0, 1]
            ])
            mesh.apply_transform(matrix_transform)

            scene_objects.append(mesh)

            print(f"   ✅ Стена #{i + 1}: {len(scaled_cm)} точек, "
                  f"размеры: {scaled_cm[:, 0].max() - scaled_cm[:, 0].min():.1f}×"
                  f"{scaled_cm[:, 1].max() - scaled_cm[:, 1].min():.1f} см")

        except Exception as e:
            print(f"   ❌ Ошибка стены #{i + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not scene_objects:
        raise ValueError("Не удалось создать ни одной стены!")

    # Создаем пол
    if all_cont:
        all_cont = np.vstack(all_cont)
        hull = ConvexHull(all_cont)
        floor_contour = all_cont[hull.vertices]
        floor_contour = np.vstack([floor_contour, floor_contour[0]])
        scene_objects.append(gen_floor(floor_contour))

    return trimesh.Scene(scene_objects)