import os
from PIL import Image

# Папка с исходными файлами
INPUT_DIR = r"kristal"
# Папка для сохранения результата
OUTPUT_DIR = r"kristal1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(input_path, output_path):
    img = Image.open(input_path).convert("RGB")   # RGBA, чтобы не сломать прозрачность
    pixels = img.load()

    width, height = img.size

    # for y in range(height):
    #     for x in range(width):
    #         r, g, b = pixels[x, y]
    #
    #         # Сначала: черный → белый
    #         if r == 0 and g == 0 and b == 0:
    #             pixels[x, y] = (255, 255, 255)
    #             continue  # уже заменили, дальше не проверяем
    #
    # # for y in range(height):
    # #     for x in range(width):
    # #         r, g, b = pixels[x, y]
    #         # Потом: зеленый → черный
    #         if r == 83 and g == 94 and b == 76:
    #             pixels[x, y] = (0, 0, 0)
    #         if r == 137 and g == 150 and b == 130:
    #             pixels[x, y] = (0, 0, 0)
    #         if r == 180 and g == 188 and b == 176:
    #             pixels[x, y] = (0, 0, 0)
    #         if r == 218 and g == 221 and b == 215:
    #             pixels[x, y] = (0, 0, 0)

    # === 1. Заменяем чистый чёрный на белый ===
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r == 0 and g == 0 and b == 0:     # точный чёрный
                pixels[x, y] = (255, 255, 255)   # заменяем на белый

    # === 2. Превращаем всё изображение в оттенки серого ===
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            # Чёрный уже заменён на белый — оставляем как есть
            if (r, g, b) == (255, 255, 255):
                continue

            # стандартная формула яркости (ITU-R BT.601)
            gray = int(0.299*r + 0.587*g + 0.114*b)
            pixels[x, y] = (gray, gray, gray)

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r == 143 and g == 143 and b == 143:
                pixels[x, y] = (0, 0, 0)
            if r == 142 and g == 142 and b == 142:
                pixels[x, y] = (0, 0, 0)
            if r == 88 and g == 88 and b == 88:
                pixels[x, y] = (0, 0, 0)

    img.save(output_path)

# Обход всех файлов в папке
for filename in os.listdir(INPUT_DIR):
    lower = filename.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Processing {in_path} -> {out_path}")
        process_image(in_path, out_path)

print("Готово.")
