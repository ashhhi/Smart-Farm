import cv2
import numpy as np

# 读取图像
image = cv2.imread('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.jpg', 0)

# 定义参数
center_pixel = (100, 100)  # 中心像素点的坐标 (i, j)
segment_length = 55  # 线段长度
orientation_step = 5  # 方向间隔（步长）
num_filters = 36  # 滤波器数量

# 计算每个方向上线段的平均灰度值
average_gray_levels = []

for angle in range(0, 180, orientation_step):
    # 计算线段的起始和结束点坐标
    start_point = (center_pixel[0] - int(segment_length / 2), center_pixel[1])
    end_point = (center_pixel[0] + int(segment_length / 2), center_pixel[1])

    # 创建线段
    matrix = cv2.getRotationMatrix2D(center_pixel, angle, 1)
    rotated_image = cv2.warpAffine(image, matrix, image.shape[::-1])
    segment = rotated_image[start_point[1], start_point[0]:end_point[0]]

    # 计算线段上的平均灰度值
    average_gray_level = np.mean(segment)
    average_gray_levels.append(average_gray_level)

# 打印每个方向上线段的平均灰度值
for i, average_gray_level in enumerate(average_gray_levels):
    print(f"Angle: {i * orientation_step} degrees, Average Gray Level: {average_gray_level}")