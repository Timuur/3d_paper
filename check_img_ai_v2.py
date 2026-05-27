from ultralytics import YOLO
import os
import sys
import cv2
from gen_mod1 import get_file_path

import time
from functools import wraps

def timing(func):
    """Декоратор для замера времени выполнения функции"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"⏱ {func.__name__}: {(t1-t0)*1000:.1f} ms" if (t1-t0) < 1 else f"⏱ {func.__name__}: {t1-t0:.2f} sec")
        return result
    return wrapper

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Глобальный кэш модели
_model_cache = None
_labels_cache = None
_obj_path_ai = get_file_path('ai_model/my_model.pt')
_device_cache = None  # Кэш выбранного устройства



def _get_device():
    """Проверка доступности CUDA и выбор устройства (с кэшированием)"""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if HAS_TORCH and torch.cuda.is_available():
        _device_cache = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Обнаружена CUDA: {gpu_name}")
    else:
        _device_cache = 'cpu'
        print("⚠️ CUDA не доступна, используется CPU")

    return _device_cache


def _load_model():
    """Ленивая загрузка модели с кэшированием и явным выбором устройства"""
    global _model_cache, _labels_cache

    if _model_cache is None:
        if not os.path.exists(_obj_path_ai):
            print('❌ WARNING: Model path is invalid or model was not found.')
            sys.exit(1)

        # device = "cpu"
        device = _get_device()

        # 1. Загружаем модель стандартным способом
        _model_cache = YOLO(_obj_path_ai, task='detect')

        # 2. Явно переносим модель на выбранное устройство (стандартный PyTorch-способ)
        _model_cache.to(device)

        _labels_cache = _model_cache.names
        print(f"✅ Модель загружена: {_obj_path_ai} на устройстве: {device}")

    return _model_cache, _labels_cache

@timing
def get_coord(img):
    """Получение детекций с использованием кэшированной модели"""
    model, labels = _load_model()

    frame = cv2.imread(img)
    if frame is None:
        raise ValueError(f"Не удалось загрузить изображение: {img}")

    # Модель уже на нужном устройстве благодаря .to(device) в _load_model
    results = model.track(frame, verbose=False)
    return results[0].boxes, labels


def clear_cache():
    """Полный сброс кэшей модуля"""
    global _model_cache, _labels_cache, _device_cache

    # Удаляем ссылку на модель (обязательно для освобождения VRAM)
    if _model_cache is not None:
        del _model_cache
        _model_cache = None

    if _labels_cache is not None:
        _labels_cache = None

    # Сбрасываем кэш устройства, чтобы при следующей загрузке он определился заново
    _device_cache = None

    # Принудительный сборщик мусора
    import gc
    gc.collect()

    # Очистка кэша PyTorch CUDA
    try:
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Ошибка при очистке CUDA кэша: {e}")

    print(f"🗑 [{time.strftime('%H:%M:%S')}] Кэш модели check_img_ai очищен")