import cv2
import numpy as np
import matplotlib.pyplot as plt
from adaptiveFilter_Conv import adaptiveDirectionFilter
from Model.Crop_Segmentation.Utils.VegetationIndices import VegetationIndices


def hysteresisThresh(image, TH=170, TL=100):
    # 滞后性阈值
    res = np.zeros(image.shape)  # 定义双阈值图像
    # 关键在这两个阈值的选择
    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            if image[i, j] < TL:
                res[i, j] = 0
            elif image[i, j] > TH:
                res[i, j] = 255
            elif ((image[i + 1, j] > TH) or (image[i - 1, j] > TH) or (image[i, j + 1] > TH) or
                  (image[i, j - 1] > TH) or (image[i - 1, j - 1] > TH) or (image[i - 1, j + 1] > TH) or
                  (image[i + 1, j + 1] > TH) or (image[i + 1, j - 1] > TH)):
                res[i, j] = 255
    return res

# 读取图像并转换为灰度图像
image = cv2.imread('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Handle/7.5 iphone/IMG_7748.jpeg')
image = cv2.resize(image, (320, 320))
gray_image = (VegetationIndices(image).ExG() + 1) * 127.5
# 计算最大灰度值
max_value = np.max(gray_image)
# 灰度拉伸
gray_image = gray_image.astype(float) / max_value * 255
# adaptive = adaptiveDirectionFilter(gray_image, 11, 1, 5)

# 对灰度图像进行高斯平滑
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)


adaptive = adaptiveDirectionFilter(blurred_image, 5, 1, 5)

thresh = hysteresisThresh(adaptive, TH=70, TL=50)

# 浮点型转成uint8型
thresh = cv2.convertScaleAbs(thresh)


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
plt.title("Adaptive Direction Filter")
plt.imshow(adaptive, cmap='gray')


plt.subplot(2, 4, 5)
plt.title("Hysteresis Threshed Image")
plt.imshow(thresh, cmap='gray')

plt.subplot(2, 4, 6)
plt.title("Largest Edge Contour")
plt.imshow(result, cmap='gray')


plt.show()