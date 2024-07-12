import cv2
import numpy as np

# 读取图片
img = cv2.imread('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/XIAOMICamera/6_29/PIC_20240629_144416410.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([img], [1], None, [256], [0, 256])

# sobel边缘检测
edges = cv2.Sobel(gray, cv2.CV_16S, 1, 1)
# 浮点型转成uint8型
edges = cv2.convertScaleAbs(edges)

# 对边缘图像进行阈值化处理
_, binary_edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# 开闭操作
iterations = 1
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(binary_edges,cv2.MORPH_CLOSE, kernel, iterations=iterations)
opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)


contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# long_contours = list(filter(lambda c: len(c) > 300, contours))

largest_contour = max(contours, key=lambda arr: len(arr))

result = np.zeros_like(img)
cv2.drawContours(result, largest_contour, -1, (0, 255, 0), thickness=cv2.FILLED)

# # 创建一个新的彩色图像
# result = np.zeros_like(cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2BGR))
#
# # 将黑色区域变为绿色
# result[binary_edges == 255] = (0, 255, 0)
#
# # 将白色区域变为蓝色
# result[binary_edges == 0] = (255, 0, 0)

cv2.imshow('Segmented Image', opening)



cv2.waitKey(0)
cv2.destroyAllWindows()