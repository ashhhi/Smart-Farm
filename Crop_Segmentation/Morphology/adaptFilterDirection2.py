import cv2
import numpy as np
from tqdm import tqdm

# 读取图像
origin_image = cv2.imread('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png', 0)

# image_size -> (240, 320)
image = cv2.resize(origin_image, (320, 240))

image_draw = image.copy()
image_draw = cv2.cvtColor(image_draw, cv2.COLOR_GRAY2BGR)

# 定义参数
# center_pixel = (106, 140)  # 中心像素点的坐标 (i, j)
segment_length = 20  # 线段长度
orientation_step = 5  # 方向间隔（步长）
num_filters = 36  # 滤波器数量
flag = 0

for i in tqdm(range(0, image.shape[0], 5)):
    for j in range(0, image.shape[1], 5):
        if image[i][j] == 0:
            continue
        center_pixel = (i, j)
        # 计算每个方向上线段的平均灰度值
        average_gray_levels = []

        for angle in range(0, 180, orientation_step):
            # 计算线段的起始和结束点坐标
            start_point = (center_pixel[0], center_pixel[1] - int(segment_length / 2))
            end_point = (center_pixel[0], center_pixel[1] + int(segment_length / 2))

            # 创建线段
            matrix = cv2.getRotationMatrix2D(center_pixel, angle, 1)
            rotated_image = cv2.warpAffine(image, matrix, image.shape[::-1])
            segment = rotated_image[center_pixel[0], max(0, start_point[1]): min(image.shape[1]-1, end_point[1])]
            # print(image[center_pixel[1]][center_pixel[0]], rotated_image[center_pixel[1]][center_pixel[0]])

            # 计算线段上的平均灰度值
            average_gray_level = np.mean(segment)
            average_gray_levels.append(average_gray_level)

        max_gray_level = max(average_gray_levels)
        angle = 5 * average_gray_levels.index(max_gray_level)
        print('方向角度', angle)


        # 在原图上绘制直线 画图横纵坐标要倒转
        line_length = segment_length  # 直线长度
        line_angle = np.deg2rad(angle)  # 将角度转换为弧度
        # line_angle = np.deg2rad(0)  # 将角度转换为弧度
        line_endpoint_x = int(center_pixel[0] + line_length * np.cos(line_angle))
        line_endpoint_y = int(center_pixel[1] - line_length * np.sin(line_angle))
        cv2.line(image_draw, (center_pixel[1], center_pixel[0]), (line_endpoint_y, line_endpoint_x), (0, 0, 255))  # 绘制直线
        # 在原图上绘制中心点
        circle_radius = 1  # 圆点半径
        cv2.circle(image_draw, (center_pixel[1], center_pixel[0]), circle_radius, (0, 255, 0), -1)  # 绘制实心圆点


# 显示结果图像
cv2.imshow("Image with Line and Center Point", image_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()