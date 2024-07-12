import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/XIAOMICamera/6_29/PIC_20240629_144416410.jpg')

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算灰度图像的直方图
gray_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 显示灰度图像及其直方图
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title("Gray Image")
plt.imshow(gray_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Gray Histogram")
plt.plot(gray_hist)
plt.xlim([0, 256])

# 计算彩色图像的直方图
colors = ('b', 'g', 'r')
plt.subplot(2, 2, 3)
plt.title("Color Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.title("Color Histogram")
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()


