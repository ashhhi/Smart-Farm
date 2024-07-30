import cv2
import numpy as np
import matplotlib.pyplot as plt
from adaptiveFilter_Conv import adaptiveDirectionFilter


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
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 计算最大灰度值
max_value = np.max(gray_image)
# 灰度拉伸
gray_image = gray_image.astype(float) / max_value * 255
# adaptive = adaptiveDirectionFilter(gray_image, 11, 1, 5)

# 对灰度图像进行高斯平滑
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)


adaptive = adaptiveDirectionFilter(blurred_image, 5, 1, 5)

thresh = hysteresisThresh(adaptive, TH=100, TL=40)

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

# 保存结果
cv2.imwrite('original_image.png', image)
cv2.imwrite('gray_image.png', gray_image)
cv2.imwrite('blurred_image.png', blurred_image)
cv2.imwrite('ADF_image.png', adaptive)
cv2.imwrite('thresh_image.png', thresh)
cv2.imwrite('result.png', result)
# 保存结果图像
plt.figure(figsize=(15, 7))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(adaptive, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(result, cmap='gray')
plt.axis('off')