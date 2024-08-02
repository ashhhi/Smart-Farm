import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
image = cv2.imread('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Handle/7.5 iphone/IMG_7748.jpeg')
image = cv2.resize(image, (320, 320))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 计算最大灰度值
max_value = np.max(gray_image)
# 灰度拉伸
gray_image = gray_image.astype(float) / max_value * 255

# 对灰度图像进行高斯平滑
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

# 边缘检测
edges = cv2.Sobel(blurred_image, cv2.CV_16S, 1, 1)

# 浮点型转成uint8型
edges = cv2.convertScaleAbs(edges)

# thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.threshold(edges, 5, 255, cv2.THRESH_BINARY)[1]
# 反转阈值结果
# thresh = cv2.bitwise_not(thresh[1])

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

dilated_image = cv2.dilate(thresh, kernel, iterations=6)


tmp = thresh
contours, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# long_contours = list(filter(lambda c: len(c) > 100, contours))

# largest_contour = max(contours, key=lambda arr: len(arr))
# 面积过滤器
filtered_contours = []
for contour in contours:
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    if area > 50000:  # 调整阈值以筛选出感兴趣的轮廓
        filtered_contours.append(contour)


result = np.zeros_like(image)
cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), thickness=cv2.FILLED)
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), thickness=3)

# 显示结果
plt.figure(figsize=(15, 7))

plt.subplot(2, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 4, 2)
plt.title("Gray Image")
plt.imshow(gray_image, cmap='gray')

plt.subplot(2, 4, 3)
plt.title("Blurred Image")
plt.imshow(blurred_image, cmap='gray')

plt.subplot(2, 4, 4)
plt.title("Edge Detection Image")
plt.imshow(edges, cmap='gray')

plt.subplot(2, 4, 5)
plt.title("Global Threshed Image")
plt.imshow(thresh, cmap='gray')

# plt.subplot(2, 4, 6)
# plt.title("Dilated Image")
# plt.imshow(dilated_image, cmap='gray')


plt.subplot(2, 4, 6)
plt.title("Largest Edge Contour")
plt.imshow(result, cmap='gray')

plt.show()