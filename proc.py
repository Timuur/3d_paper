import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('prime/4p.png')

# Конвертация в grayscale (удаление цветов)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Бинаризация изображения
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Создание ядра для морфологических операций
kernel = np.ones((3, 3), np.uint8)

# Удаление тонких линий (морфологическое открытие)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Инверсия обратно к исходным цветам
result = cv2.bitwise_not(cleaned)

# Сохранение результата
# cv2.imwrite('output.png', result)
cv2.imshow("Result", result)
cv2.waitKey(0)