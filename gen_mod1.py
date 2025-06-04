import cv2
import numpy as np
import trimesh
from tabulate import tabulate
import img2wall as i2w
from scipy.spatial import ConvexHull
from trimesh.transformations import (rotation_matrix,scale_matrix,
                                     concatenate_matrices)
from shapely.geometry import Polygon

mesh_door = trimesh.load('Door_Component.obj')

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

def build_door(wall_contours, scale):
    scene_objects = []
    for i, contour in enumerate(wall_contours):
        mesh = mesh_door.copy()
        wight_d = contour[2][0] - contour[0][0]
        hight_d = contour[1][1] - contour[0][1]
        contour = i2w.average_close_points(contour, 300)
        contour = np.column_stack((contour, np.zeros(len(contour))))

        # print(contour)
        # Преобразуем контур в массив точек
        contour_points = np.array(contour[0])

        # Масштабируем координаты
        scaled = contour_points * float(scale)
        # print(scaled)
        # print(scaled[0])
        # print(translation_matrix(scaled))

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        try:
            matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), 0  # Устанавливаем смещение
            # matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1] + (hight_d * float(scale))/2), 0  # Устанавливаем смещение
            # Создаём 2D полигон
            mesh.apply_scale(6)
            rot = rotation_matrix(np.pi / 2, [-1, 0, 0])
            transform = concatenate_matrices(matrix_z_inversion, rot)
            mesh.apply_transform(transform)
            # print(transform)
            if wight_d < hight_d:
                # 1. Находим его центр (bounding box центроид)
                center = mesh.bounding_box.centroid

                # 2. Переносим в начало координат
                mesh.apply_translation(-center)

                # 3. Поворачиваем (например, на 45° вокруг оси Z)
                rotation = rotation_matrix(np.pi / 2, [0, 0, 1])  # Ось Z
                mesh.apply_transform(rotation)

                # 4. Возвращаем на исходную позицию
                mesh.apply_translation(center)

            # Добавляем в сцену
            scene_objects.append(mesh)

            print(f"Контур DOOR #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура DOOR #{i + 1}: {str(e)}")
            continue

    return scene_objects


def create_rectangular_window(width=1.5, height=1.0, frame_width=0.1, frame_thickness=0.05,
                              sill_width=0.2, sill_thickness=0.05, sill_protrusion=0.15,
                              glass_thickness=0.01):
    """
    Creates a 3D model of a rectangular window with a windowsill

    Parameters:
    width: window width
    height: window height
    frame_width: window frame width
    frame_thickness: window frame thickness
    sill_width: windowsill width
    sill_thickness: windowsill thickness
    sill_protrusion: windowsill protrusion from the wall
    glass_thickness: glass thickness

    Returns:
    Combined trimesh.Trimesh object containing all window elements
    """
    # Create a list to store all window components
    components = []

    # 1. Create window frame (4 parts instead of boolean difference)
    # Vertical frame parts
    left_frame = trimesh.primitives.Box(
        extents=[frame_width, frame_thickness, height + 2 * frame_width])
    right_frame = trimesh.primitives.Box(
        extents=[frame_width, frame_thickness, height + 2 * frame_width])

    # Horizontal frame parts
    top_frame = trimesh.primitives.Box(
        extents=[width, frame_thickness, frame_width])
    bottom_frame = trimesh.primitives.Box(
        extents=[width, frame_thickness, frame_width])

    # Position frame parts
    left_frame.apply_translation([-(width / 2 + frame_width / 2), 0, 0])
    right_frame.apply_translation([(width / 2 + frame_width / 2), 0, 0])
    top_frame.apply_translation([0, 0, (height / 2 + frame_width / 2)])
    bottom_frame.apply_translation([0, 0, -(height / 2 + frame_width / 2)])

    # 2. Create glass
    glass = trimesh.primitives.Box(
        extents=[width - 0.02, glass_thickness, height - 0.02])
    glass.apply_translation([0, frame_thickness / 2, 0])

    # 3. Create windowsill
    sill = trimesh.primitives.Box(
        extents=[width + 2 * sill_protrusion,
                 sill_width,
                 sill_thickness])
    sill.apply_translation([0,
                            -frame_thickness / 2 - sill_width / 2,
                            height / 2 - frame_width - sill_thickness / 2])

    # Convert all primitives to Trimesh objects
    components = [
        left_frame, right_frame, top_frame, bottom_frame,
        glass, sill
    ]
    meshes = [m if isinstance(m, trimesh.Trimesh) else m.to_mesh() for m in components]

    # Verify all meshes are watertight
    for mesh in meshes:
        if not mesh.is_volume:
            # Create a new mesh by copying vertices and faces
            repaired = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                                       faces=mesh.faces.copy())
            repaired.fill_holes()
            repaired.fix_normals()
            meshes[meshes.index(mesh)] = repaired

    # Combine all components
    window_mesh = trimesh.util.concatenate(meshes)

    return window_mesh

def build_window(window_position, wall_contours, scale, height = 2.7):
    scene_objects = []
    # for i, contour in enumerate(wall_contours):
    #     for j, point in enumerate(contour):

    for i, contour in enumerate(window_position):
        wight_d = contour[2][0] - contour[0][0]
        hight_d = contour[1][1] - contour[0][1]
        print(contour)
        # print(contour * float(scale))
        # print(translation_matrix(scaled))
        wight_w = wight_d * float(scale)
        print(wight_w)
        print(wight_w * float(scale))

        mesh = create_rectangular_window(wight_w/2, 1.5)

        matrix_z_inversion = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        contour = i2w.average_close_points(contour, 300)
        contour = np.column_stack((contour, np.zeros(len(contour))))

        # print(contour)
        # Преобразуем контур в массив точек
        contour_points = np.array(contour[0])

        # Масштабируем координаты
        scaled = contour_points * float(scale)
        print(scaled)
        try:
            matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1]), float(height)*0.5  # Устанавливаем смещение
            # matrix_z_inversion[:3, 3] = scaled[0], -(scaled[1] + (hight_d * float(scale))/2), 0  # Устанавливаем смещение
            # Создаём 2D полигон
            mesh.apply_scale(6)
            # rot = rotation_matrix(np.pi / 2, [0, -1, 0])
            # transform = concatenate_matrices(matrix_z_inversion, rot)
            transform = concatenate_matrices(matrix_z_inversion)
            mesh.apply_transform(transform)
            # print(transform)
            if wight_d < hight_d:
                # 1. Находим его центр (bounding box центроид)
                center = mesh.bounding_box.centroid

                # 2. Переносим в начало координат
                mesh.apply_translation(-center)

                # 3. Поворачиваем (например, на 45° вокруг оси Z)
                rotation = rotation_matrix(np.pi / 2, [0, 0, 1])  # Ось Z
                mesh.apply_transform(rotation)

                # 4. Возвращаем на исходную позицию
                mesh.apply_translation(center)

            # Добавляем в сцену
            scene_objects.append(mesh)

            print(f"Контур WIN #{i + 1} успешно преобразован в 3D-объект")

        except Exception as e:
            print(f"Ошибка обработки контура WINDOQ #{i + 1}: {str(e)}")
            continue
    print("window")

    return scene_objects

def build_3d_model(wall_contours, original_size, scale=0.1, height=3.0):
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
    all_cont = []

    for i, contour in enumerate(wall_contours):
        # Преобразуем контур в массив точек
        contour_points = np.array(contour).reshape(-1, 2)

        # Упрощаем контур
        contour_points = contour_points.reshape(-1, 1, 2).astype(np.int32).squeeze()

        # Пропускаем контуры с малым количеством точек
        if len(contour_points) < 3:
            print(f"Контур #{i + 1} пропущен (мало точек: {len(contour_points)})")
            continue

        # next_wall = wall_contours[(i + 1) % len(wall_contours)]

        # build_door(wall_contours[i], next_wall, scale)

        # Масштабируем координаты
        scaled = contour_points * float(scale)
        print(scaled)
        all_cont.append(scaled)


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
            # print("стена = ",scaled)

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

    print(all_cont)
    all_cont = np.vstack(all_cont)

    # Находим выпуклую оболочку
    hull = ConvexHull(all_cont)

    # Получаем точки выпуклой оболочки (уже упорядочены)
    kr_p = all_cont[hull.vertices]

    # Замыкаем контур (добавляем первую точку в конец)
    kr_p = np.vstack([kr_p, kr_p[0]])

    scene_objects.append(gen_floor(kr_p))

    return trimesh.Scene(scene_objects)