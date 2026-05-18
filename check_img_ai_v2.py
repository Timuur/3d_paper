from ultralytics import YOLO
import os
import sys
import cv2
from gen_mod1 import get_file_path

#Глобальный кэш модели
_model_cache = None
_labels_cache = None
_obj_path_ai = get_file_path('ai_model/my_model.pt')


def _load_model():
    """Ленивая загрузка модели с кэшированием"""
    global _model_cache, _labels_cache

    if _model_cache is None:
        if not os.path.exists(_obj_path_ai):
            print('WARNING: Model path is invalid or model was not found.')
            sys.exit()
        _model_cache = YOLO(_obj_path_ai, task='detect')
        _labels_cache = _model_cache.names
        print(f"✅ Модель загружена: {_obj_path_ai}")

    return _model_cache, _labels_cache


def get_coord(img):
    """Получение детекций с использованием кэшированной модели"""
    model, labels = _load_model()  # ← Используем кэш
    frame = cv2.imread(img)
    results = model.track(frame, verbose=False)
    return results[0].boxes, labels


def clear_cache():
    """Полный сброс кэшей модуля"""
    global _model_cache, _labels_cache

    # 1. Удаляем ссылку на модель (освобождение памяти)
    if _model_cache is not None:
        del _model_cache
        _model_cache = None

    if _labels_cache is not None:
        _labels_cache = None

    # 2. Принудительный сборщик мусора (важно для CUDA-тензоров)
    import gc
    gc.collect()

    # 3. Если используется CUDA — очистка кэша PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

    print("🗑 Кэш модели check_img_ai очищен")