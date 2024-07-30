import cv2
import numpy as np
import matplotlib.pyplot as plt

def removebackground(images):
    res = []
    # 读取图像并转换为灰度图像
    for image in images:
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


        tmp = thresh
        contours, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # long_contours = list(filter(lambda c: len(c) > 100, contours))

        max_contour = max(contours, key=lambda arr: len(arr))


        # 创建与原始图像相同大小的空白图像
        mask = np.zeros_like(gray_image)

        # 在空白图像上绘制最大轮廓
        mask = cv2.drawContours(mask, [max_contour], -1, 1, thickness=cv2.FILLED)

    res.append(mask)
    res = np.array(res)
    return res

# removebackground('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Handle/7.5 iphone/IMG_7748.jpeg')