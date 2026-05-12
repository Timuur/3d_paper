import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, concatenate_matrices, translation_matrix
from trimesh.boolean import difference, union
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union, snap
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─── Константы ─────────────────────────────────────────────────────────────
BOOL_EPS = 0.02  # Увеличенный допуск для булевых операций
WALL_THICKNESS = 0.25  # Толщина стены в метрах
OPENING_TOLERANCE = 0.15  # Допуск для привязки проёма к стене (в метрах)
MIN_OPENING_SIZE = 0.3  # Минимальный размер проёма, чтобы его учитывать


def get_file_path(filename: str) -> Path:
    base_path = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).parent
    return base_path / filename


# ─── Загрузка мешей с проверкой ────────────────────────────────────────────
def _load_mesh_safe(path: Path, default_scale: float = 8.0):
    """Безопасная загрузка меша с валидацией."""
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh.is_empty:
            logger.error(f"❌ Пустой меш: {path}")
            return None
        mesh.apply_scale(default_scale)
        mesh.fix_normals()
        logger.info(f"✅ Загружен меш: {path.name}, вершин: {len(mesh.vertices)}")
        return mesh
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки {path}: {e}")
        return None


_DOOR_MESH = _load_mesh_safe(get_file_path("3d_obj_test/Door_Component.obj"))
_WIN_MESH = _load_mesh_safe(get_file_path("3d_obj_test/window1.obj"))
_FRI_MESH = _load_mesh_safe(get_file_path("3d_obj_test/Separate_assets_obj/fridge_001.obj"))
_KitSink_MESH = _load_mesh_safe(get_file_path("3d_obj_test/Separate_assets_obj/kitchen_sink_001.obj"))
_WasMach_MESH = _load_mesh_safe(get_file_path("3d_obj_test/Separate_assets_obj/washing_machine_001.obj"))


def _create_wall_polygon_from_line(line_pts: np.ndarray, thickness: float, scale: float) -> Polygon:
    """
    Создаёт полигон стены с толщиной из линии (2 точки) или контура.
    line_pts: массив точек [[x1,y1], [x2,y2], ...] в пикселях
    """
    pts_scaled = line_pts * scale

    if len(pts_scaled) == 2:
        # Простой отрезок — создаём прямоугольник с толщиной
        p1, p2 = pts_scaled[0], pts_scaled[1]
        # Вектор направления
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        direction = direction / length
        # Нормаль (перпендикуляр)
        normal = np.array([-direction[1], direction[0]])
        # 4 угла прямоугольника
        half_t = thickness / 2
        corners = [
            p1 + normal * half_t,
            p1 - normal * half_t,
            p2 - normal * half_t,
            p2 + normal * half_t,
        ]
        poly = Polygon(corners)
    else:
        # Контур из 4+ точек — берём выпуклую оболочку и добавляем толщину
        poly = Polygon(pts_scaled)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # Добавляем "толщину" через buffer, если полигон слишком тонкий
        if poly.area < 1e-4:
            poly = poly.buffer(thickness / 2, cap_style=2, join_style=2)

    return poly if poly.is_valid and poly.area > 1e-4 else None


def _create_wall_mesh_3d(polygon: Polygon, height: float) -> Optional[trimesh.Trimesh]:
    """Создаёт 3D меш стены из 2D полигона."""
    try:
        mesh = trimesh.creation.extrude_polygon(polygon, height=height)
        # Корректировка системы координат: Y вниз → Y вверх
        flip = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        mesh.apply_transform(flip)
        mesh.fix_normals()
        # Цвет для отладки — стены синие
        mesh.visual.face_colors = [100, 100, 255, 255]
        return mesh
    except Exception as e:
        logger.error(f"❌ Ошибка создания меша стены: {e}")
        return None


def _parse_opening_contour(contour, scale: float) -> dict:
    """Парсит контур проёма и возвращает структурированные данные."""
    pts = np.array(contour, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        logger.warning(f"⚠️ Контур проёма имеет {len(pts)} точек, ожидаем 4")
        return None

    # Центр и размеры в метрах
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    width_px, height_px = x_max - x_min, y_max - y_min
    width_m, height_m = width_px * scale, height_px * scale

    if width_m < MIN_OPENING_SIZE or height_m < MIN_OPENING_SIZE:
        logger.warning(f"⚠️ Проём слишком мал: {width_m:.2f}x{height_m:.2f}м")
        return None

    center_px = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    center_m = center_px * scale

    # Угол поворота
    edges = np.diff(pts, axis=0, append=pts[0].reshape(1, -1))
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    longest_idx = np.argmax(lengths)
    angle = np.arctan2(edges[longest_idx, 1], edges[longest_idx, 0])

    return {
        'polygon': Polygon(pts * scale),
        'center_2d': center_m,
        'center_3d': None,  # Заполнится позже
        'width': width_m,
        'height': height_m,
        'angle': angle,
        'bbox': box(x_min * scale, y_min * scale, x_max * scale, y_max * scale)
    }

# ─── НОВАЯ УСТОЙЧИВАЯ ФУНКЦИЯ ПРИВЯЗКИ ─────────────────────────────────────
def _find_nearest_wall(opening: dict, wall_meshes: List[trimesh.Trimesh], max_dist: float= 5.0) -> Optional[int]:
    """
    Привязывает проём к стене по 2D bbox в единой системе координат.
    Работает быстрее и стабильнее Shapely.distance().
    """
    cx, cy = opening['center_2d']  # Координаты в пиксельной системе (Y вниз)

    # Преобразуем в систему trimesh (Y инвертирован)
    cx_3d, cy_3d = cx, -cy
    best_idx = None
    best_dist = float('inf')

    for i, wall in enumerate(wall_meshes):
        wb = wall.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

        # Расстояние от точки до AABB (bounding box)
        dx = max(wb[0, 0] - cx_3d, 0, cx_3d - wb[1, 0])
        dy = max(wb[0, 1] - cy_3d, 0, cy_3d - wb[1, 1])
        dist = np.hypot(dx, dy)

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_dist <= max_dist:
        logger.info(f"✅ Проём привязан к Стене #{best_idx + 1} (смещение: {best_dist:.2f}м)")
        return best_idx

    logger.warning(f"⚠️ Проём не привязан: центр=({cx:.1f}, {cy:.1f}), "
                   f"ближайшая стена в {best_dist:.2f}м (лимит: {max_dist}м)")
    # for i, wall in enumerate(wall_meshes):
    #     wb = wall.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    #
    #     # Проверяем попадание по X и Y с допуском
    #     in_x = (wb[0, 0] - OPENING_TOLERANCE <= cx_3d <= wb[1, 0] + OPENING_TOLERANCE)
    #     in_y = (wb[0, 1] - OPENING_TOLERANCE <= cy_3d <= wb[1, 1] + OPENING_TOLERANCE)
    #
    #     if in_x and in_y:
    #         logger.debug(f"✅ Дверь/Окно привязано к Стене #{i + 1} (центр={cx:.1f},{cy:.1f})")
    #         return i
    #
    # # Если не нашли, выводим отладку
    # logger.warning(f"⚠️ Проём не привязан: центр=({cx:.1f}, {cy:.1f}) -> 3D=({cx_3d:.1f}, {cy_3d:.1f})")
    # for i, w in enumerate(wall_meshes):
    #     logger.warning(f"   Стена #{i + 1} bbox X:[{w.bounds[0, 0]:.1f}, {w.bounds[1, 0]:.1f}] "
    #                    f"Y:[{w.bounds[0, 1]:.1f}, {w.bounds[1, 1]:.1f}]")
    return None


def _create_opening_mesh(opening: dict, base_mesh: trimesh.Trimesh,
                         room_height: float, is_window: bool, is_door: bool) -> Optional[trimesh.Trimesh]:
    """Создаёт 3D меш двери/окна с правильной позицией и ориентацией."""
    if base_mesh is None:
        return None

    obj = base_mesh.copy()

    # Масштабирование под размер проёма
    obj_bounds = obj.bounds
    obj_size = obj_bounds[1] - obj_bounds[0]

    # Выбираем ось для масштабирования по большей стороне проёма
    if opening['width'] >= opening['height']:
        scale_factor = opening['width'] / (obj_size[0] + 1e-8)
    else:
        scale_factor = opening['height'] / (obj_size[2] + 1e-8)

    obj.apply_scale(scale_factor)

    # Поворот из плоскости XY в вертикальную плоскость стены
    rot_to_wall = rotation_matrix(np.pi / 2, [-1, 0, 0])  # Поворот вокруг X
    obj.apply_transform(rot_to_wall)

    # Позиция по Z: окно — по центру высоты, дверь — снизу
    if is_window:
        z_pos = room_height / 2.0
        # Поворот по углу контура (в плоскости XY)
        if abs(opening['angle']) > 0.01:
            rot_z = rotation_matrix(opening['angle'], [0, 0, 1])
            obj.apply_transform(rot_z)
        rot_vertical = rotation_matrix(np.pi, [1, 0, 0])
        obj.apply_transform(rot_vertical)
        rot_fix = rotation_matrix(np.pi/2, [0, 0, -1])
        obj.apply_transform(rot_fix)
    else:
        if is_door:
            z_pos = 0  # Дверь стоит на полу
            # z_pos = opening['height'] / 2.0  # Дверь стоит на полу
            # Поворот по углу контура (в плоскости XY)
            if abs(opening['angle']) > 0.01:
                rot_z = rotation_matrix(opening['angle'], [0, 0, 1])
                obj.apply_transform(rot_z)
            rot_vertical = rotation_matrix(np.pi, [1, 0, 0])
            obj.apply_transform(rot_vertical)


    # Финальная позиция: инвертируем Y для trimesh
    center_3d = np.array([
        opening['center_2d'][0],
        -opening['center_2d'][1],  # Инверсия Y!
        z_pos
    ])
    opening['center_3d'] = center_3d  # Сохраняем для отладки

    trans = translation_matrix(center_3d)
    obj.apply_transform(trans)

    logger.debug(f"🔍 Ориентация двери/окна: angle={np.degrees(opening['angle']):.1f}°, bbox_local={obj.bounds}, center_3d={center_3d}")

    # Цвет для отладки: двери — зелёные, окна — жёлтые
    obj.visual.face_colors = [0, 255, 0, 255] if is_window else [0, 200, 0, 255]


    return obj


def _create_cutter_for_opening(opening: dict, wall_thickness: float) -> Optional[trimesh.Trimesh]:
    """Создаёт 3D объём для вырезания проёма из стены."""
    try:
        poly = opening['polygon']
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Экструзия на толщину стены + допуск
        cutter_height = wall_thickness + 2 * BOOL_EPS
        cutter = trimesh.creation.extrude_polygon(poly, height=cutter_height)

        # Центрирование по Z относительно позиции проёма
        z_offset = opening['center_3d'][2] - cutter_height / 2
        trans = translation_matrix([0, 0, z_offset])
        cutter.apply_transform(trans)

        # Цвет для отладки — красный (невидимый, но для отладки можно включить)
        cutter.visual.face_colors = [255, 50, 50, 100]

        return cutter
    except Exception as e:
        logger.error(f"❌ Ошибка создания cutter: {e}")
        return None


def _safe_boolean_difference(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                             operation_name: str) -> trimesh.Trimesh:
    """Безопасное вырезание проёма."""
    try:
        # Простая проверка пересечения BoundingBox
        a_min, a_max = mesh_a.bounds
        b_min, b_max = mesh_b.bounds
        overlaps = not (
                a_max[0] < b_min[0] or a_min[0] > b_max[0] or
                a_max[1] < b_min[1] or a_min[1] > b_max[1] or
                a_max[2] < b_min[2] or a_min[2] > b_max[2]
        )

        if not overlaps:
            logger.debug(f"⚠️ {operation_name}: BBox не пересекаются")
            return mesh_a

        # Пытаемся вырезать, fallback на другой движок если нужно
        result = difference([mesh_a, mesh_b], engine='manifold')
        if result.is_empty: return mesh_a
        result.fix_normals()
        return result
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}")
        return mesh_a  # Возвращаем стену без изменений при ошибке


def build_3d_model(wall_contours, scale=0.1, height=3.0,
                   door_contours: List = None, window_contours: List = None,
                   wall_thickness: float = WALL_THICKNESS,
                   merge_scene: bool = True,
                   debug_export: str = None, original_image_size: Tuple[int, int] = None):  # Путь для экспорта отладочной сцены
    """
    Построение 3D модели с отладкой и исправлениями.

    # :param debug_export: если указан, экспортирует промежуточные меши в файлы для анализа
    """

    logger.info(f"🚀 Начало построения модели: {len(wall_contours)} стен, "
                f"{len(door_contours or [])} дверей, {len(window_contours or [])} окон")

    scene_objects = []
    all_wall_points = []
    wall_meshes = []
    wall_polygons = []

    # ─── 1. Создание стен ──────────────────────────────────────────────────
    logger.info("🧱 Создаём стены...")
    for i, contour in enumerate(wall_contours):
        pts = np.asarray(contour).reshape(-1, 2).astype(np.float32)
        if len(pts) < 2:
            logger.warning(f"⚠️ Стена #{i + 1}: слишком мало точек ({len(pts)})")
            continue

        # Создаём полигон стены с толщиной
        poly = _create_wall_polygon_from_line(pts, wall_thickness, scale)
        if poly is None or poly.area < 1e-4:
            logger.warning(f"⚠️ Стена #{i + 1}: не удалось создать валидный полигон")
            continue

        # Создаём 3D меш
        mesh = _create_wall_mesh_3d(poly, height)
        if mesh is None:
            continue

        wall_meshes.append(mesh)
        wall_polygons.append(poly)
        all_wall_points.append(pts * scale)
        logger.debug(f"✅ Стена #{i + 1}: площадь={poly.area:.3f}м², вершин={len(mesh.vertices)}")

        # Отладочный экспорт
        if debug_export:
            mesh.export(f"{debug_export}_wall_{i + 1}.obj")

    if not wall_meshes:
        logger.error("❌ Критическая ошибка: не создано ни одной стены!")
        return trimesh.Scene()

    # ─── 2. Обработка дверей ───────────────────────────────────────────────
    logger.info("🚪 Обрабатываем двери...")
    door_cutters = []

    if door_contours and _DOOR_MESH:
        for i, cont in enumerate(door_contours):
            opening = _parse_opening_contour(cont, scale)
            if opening is None:
                continue

            # Находим целевую стену
            wall_idx = _find_nearest_wall(opening, wall_meshes)

            if wall_idx is None:
                logger.warning(f"⚠️ Дверь #{i + 1}: не найдена стена, координаты: {opening['center_2d']}")
                continue

            # Создаём меш двери
            door_mesh = _create_opening_mesh(opening, _DOOR_MESH, height, False, True)
            if door_mesh:
                scene_objects.append(door_mesh)

            # Создаём cutter для вырезания
            cutter = _create_cutter_for_opening(opening, wall_thickness)
            if cutter:
                door_cutters.append((cutter, wall_idx))
                logger.info(
                    f"✅ Дверь #{i + 1}: {opening['width']:.2f}x{opening['height']:.2f}м → стена #{wall_idx + 1}")

    # ─── 3. Обработка окон ─────────────────────────────────────────────────
    logger.info("🪟 Обрабатываем окна...")
    window_cutters = []

    if window_contours and _WIN_MESH:
        for i, cont in enumerate(window_contours):
            opening = _parse_opening_contour(cont, scale)
            if opening is None:
                continue

            wall_idx = _find_nearest_wall(opening, wall_meshes)
            if wall_idx is None:
                logger.warning(f"⚠️ Окно #{i + 1}: не найдена ближайшая стена, пропускаем")
                continue

            win_mesh = _create_opening_mesh(opening, _WIN_MESH, height, True, False)
            if win_mesh:
                scene_objects.append(win_mesh)

            cutter = _create_cutter_for_opening(opening, wall_thickness)
            if cutter:
                window_cutters.append((cutter, wall_idx))
                logger.info(f"✅ Окно #{i + 1}: {opening['width']:.2f}x{opening['height']:.2f}м → стена #{wall_idx + 1}")

    # ─── 4. Вырезание проёмов из стен ──────────────────────────────────────
    logger.info("✂️ Вырезаем проёмы...")

    all_cutters = door_cutters + window_cutters
    for cutter, wall_idx in all_cutters:
        if wall_idx >= len(wall_meshes):
            continue

        logger.debug(f"🔪 Вырезаем проём из стены #{wall_idx + 1}")
        wall_meshes[wall_idx] = _safe_boolean_difference(
            wall_meshes[wall_idx], cutter,
            f"Вырезание проёма (стена #{wall_idx + 1})"
        )

        # Отладочный экспорт после вырезания
        if debug_export:
            wall_meshes[wall_idx].export(f"{debug_export}_wall_{wall_idx + 1}_cut.obj")

    scene_objects.extend(wall_meshes)

    # ─── 5. Создание пола ──────────────────────────────────────────────────
    logger.info("🟦 Создаём пол...")
    if all_wall_points:
        all_pts = np.vstack(all_wall_points)
        if len(all_pts) >= 3:
            try:
                # Простой пол по bounding box всех стен
                x_min, y_min = all_pts.min(axis=0)
                x_max, y_max = all_pts.max(axis=0)
                floor_poly = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

                floor = trimesh.creation.extrude_polygon(floor_poly, height=0.15)
                # Инверсия и позиционирование пола внизу
                flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                floor.apply_transform(flip)
                floor.apply_transform(translation_matrix([0, 0, 0.075]))  # Поднимаем на половину толщины
                floor.visual.face_colors = [150, 150, 150, 255]  # Серый

                scene_objects.append(floor)
                logger.info("✅ Пол создан")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать пол: {e}")

    # ─── 6. Финализация сцены ──────────────────────────────────────────────
    logger.info(f"📦 Финализация: {len(scene_objects)} объектов в сцене")

    # Отладочный экспорт перед слиянием
    if debug_export:
        trimesh.Scene(scene_objects).export(f"{debug_export}_before_merge.glb")

    if merge_scene and scene_objects:
        try:
            logger.info("🔗 Выполняем финальное объединение...")
            merged = union(scene_objects, engine='manifold')
            merged.fix_normals()
            logger.info(f"✅ Сцена объединена: {len(merged.vertices)} вершин, watertight={merged.is_watertight}")
            return merged
        except Exception as e:
            logger.warning(f"⚠️ Объединение не удалось: {e}, возвращаем раздельную сцену")

    return trimesh.Scene(scene_objects)