import cv2
import numpy as np
import trimesh

from tabulate import tabulate

import img2wall as i2w
import gen_model as gm
import gen_mod1 as gm1

if __name__ == "__main__":
    try:
        # Обработка плана помещения
        wall_contours, image_size = i2w.process_floor_plan("test/0011.jpg")

        # Параметры моделирования
        MODEL_SCALE = 0.05 # 1 пиксель = 5 см
        WALL_HEIGHT = 20  # Высота потолков 2.7 метра
        WALL_THICKNESS = 0.001  # Толщина стен 20 см
        ESP = 0.00025

        # Создание 3D-модели
        # scene = gm.build_3d_model(
        #     wall_contours,
        #     image_size,
        #     scale=MODEL_SCALE,
        #     height=WALL_HEIGHT,
        #     thickness=WALL_THICKNESS
        # )

        # Визуализация результата
        # scene.show()

        # Создание 3D-модели
        scene1 = gm1.build_3d_model(
            wall_contours,
            image_size,
            MODEL_SCALE,
            WALL_HEIGHT,
            ESP
        )

        # Визуализация результата
        scene1.show()

        # Экспорт модели (опционально)
        scene1.export('model.obj')

    except FileNotFoundError as e:
        print(f"Ошибка загрузки файла: {e}")
    except ValueError as e:
        print(f"Ошибка обработки данных: {e}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")