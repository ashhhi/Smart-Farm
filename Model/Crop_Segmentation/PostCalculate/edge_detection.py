import cv2
import numpy as np

# 读取图片
img = cv2.imread('ice_plant_stem.jpg')

# sobel边缘检测
edges = cv2.Sobel(img, cv2.CV_16S, 1, 1)
# 浮点型转成uint8型
edges = cv2.convertScaleAbs(edges)

gray_edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

# 对边缘图像进行阈值化处理
_, binary_edges = cv2.threshold(gray_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(binary_edges, kernel, iterations=1)

# # 创建一个新的彩色图像
# result = np.zeros_like(cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2BGR))
#
# # 将黑色区域变为绿色
# result[binary_edges == 255] = (0, 255, 0)
#
# # 将白色区域变为蓝色
# result[binary_edges == 0] = (255, 0, 0)

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('edges', edges)
cv2.imshow('Binary Edges', binary_edges)
cv2.imshow('dilated', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()