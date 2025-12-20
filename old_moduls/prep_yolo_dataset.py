#!/usr/bin/env python3
"""
prep_yolo_dataset.py

Утилита для подготовки датасета под Ultralytics YOLO из набора:
- data.yaml (может содержать: path, names, train: train.txt, [val: val.txt])
- labels/train/*.txt
- train.txt (список путей к изображениям)

Функции:
1) Проверка наличия изображений и соответствующих label-файлов.
2) Очистка меток: удаление строк с неверным количеством чисел (<5 для DETECT).
3) Разделение на train/val (по умолчанию 90/10) и генерация val.txt.
4) Обновление data.yaml (добавляет val: val.txt, если не было).

Использование:
    python prep_yolo_dataset.py --root <путь к корню, где лежит data.yaml> \
                                --labels_subdir labels/train \
                                --train_txt train.txt \
                                --val_ratio 0.1

После этого можно запускать обучение:
    yolo task=detect mode=train model=yolov8n.pt data=data.yaml imgsz=1280 epochs=80
"""
import argparse
import os
from pathlib import Path

def read_lines(p: Path):
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]

def write_lines(p: Path, lines):
    p.write_text("\n".join(lines) + "\n", encoding='utf-8')

def is_image(path: str):
    return path.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'))

def clean_label_file(label_path: Path) -> int:
    """Удаляет битые строки из файла меток. Возвращает количество удалённых строк."""
    if not label_path.exists():
        return 0
    changed = 0
    out_lines = []
    for ln in label_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        # DETECT формат: class x y w h  -> минимум 5 токенов
        if len(parts) < 5:
            changed += 1
            continue
        # базовая проверка чисел
        ok = True
        try:
            int(parts[0])
            for t in parts[1:5]:
                float(t)
        except Exception:
            ok = False
        if ok:
            out_lines.append(" ".join(parts))
        else:
            changed += 1
    if changed > 0:
        label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding='utf-8')
    return changed

def default_label_for_image(img_path: Path, labels_dir: Path) -> Path:
    # Ищем label по базовому имени файла с разными расширениями .txt
    base = img_path.stem
    cand = labels_dir / f"{base}.txt"
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="G:/Pract_Prog/Pract_Prog_F/12", help="Корневой каталог, где лежит data.yaml и train.txt")
    ap.add_argument("--labels_subdir", type=str, default="labels/train", help="Путь к папке с label-файлами относительно root")
    ap.add_argument("--train_txt", type=str, default="train.txt", help="Файл со списком изображений для обучения")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Доля валидации")
    ap.add_argument("--val_txt", type=str, default="val.txt", help="Имя выходного файла валидации")
    args = ap.parse_args()

    root = Path(args.root)
    labels_dir = (root / args.labels_subdir).resolve()
    train_list_path = (root / args.train_txt).resolve()
    val_list_path = (root / args.val_txt).resolve()

    if not train_list_path.exists():
        raise SystemExit(f"[ERROR] Не найден {train_list_path}")

    img_paths = [Path(p) for p in read_lines(train_list_path) if is_image(p)]
    if not img_paths:
        raise SystemExit("[ERROR] train.txt не содержит путей к изображениям или формат неизвестен")

    print(f"[INFO] Изображений в train.txt: {len(img_paths)}")
    print(f"[INFO] Папка с метками: {labels_dir}")

    # 1) Очистка меток
    cleaned_files = 0
    removed_lines_total = 0
    for img_p in img_paths:
        lbl = default_label_for_image(Path(img_p), labels_dir)
        if lbl.exists():
            removed = clean_label_file(lbl)
            if removed > 0:
                cleaned_files += 1
                removed_lines_total += removed
    print(f"[CLEAN] Файлов с исправленными метками: {cleaned_files}, удалено битых строк: {removed_lines_total}")

    # 2) Фильтрация: оставим только те изображения, которые существуют на диске
    img_paths_existing = [p for p in img_paths if p.exists()]
    if len(img_paths_existing) != len(img_paths):
        print(f"[WARN] Найдено несуществующих путей: {len(img_paths) - len(img_paths_existing)}")
    img_paths = img_paths_existing

    # 3) Разделение на train/val (простое, без стратификации)
    n = len(img_paths)
    val_n = max(1, int(n * args.val_ratio))
    train_n = n - val_n
    train_split = img_paths[:train_n]
    val_split = img_paths[train_n:]

    # 4) Запись val.txt и обновление train.txt (опционально)
    write_lines(val_list_path, [str(p) for p in val_split])
    write_lines(train_list_path, [str(p) for p in train_split])
    print(f"[SPLIT] Train: {len(train_split)}  Val: {len(val_split)}")
    print(f"[OK] Записаны:\n  train: {train_list_path}\n  val:   {val_list_path}")

    # 5) Подсказка по data.yaml
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        txt = data_yaml.read_text(encoding='utf-8', errors='ignore').strip()
        if "val:" not in txt:
            txt += f"\nval: {val_list_path.name}\n"
            data_yaml.write_text(txt, encoding='utf-8')
            print(f"[UPDATE] Добавлен путь валидации в data.yaml: val: {val_list_path.name}")
        else:
            print("[INFO] data.yaml уже содержит 'val:' — проверьте путь.")
    else:
        print("[WARN] data.yaml не найден. Создайте по образцу Ultralytics.")

if __name__ == "__main__":
    main()
