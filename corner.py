import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('prime/4p.png')

# Конвертируем BGR в RGB для корректного отображения в matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_img, 100, .06, 10)
corners = np.intp(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img_rgb, (x, y), 3, (255, 0, 0), -1)  # Используем красный цвет (в RGB)

# Показываем изображение через matplotlib
plt.imshow(img_rgb)
#plt.imshow(gray_img)
plt.axis('off')  # Скрываем оси
plt.show()