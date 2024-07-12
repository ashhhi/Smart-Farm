from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# 读取图像
origin_image = Image.open('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/Morphology/3691716968414_-pic_jpg.rf.b8f9d2f1e596dfce7d2b80ec8dc044a5.png').convert('L')

# image_size -> (240, 320)
image = origin_image.resize((320, 240))
image_pixels = image.load()

image_copy = image.copy().convert('RGB')
image_draw = ImageDraw.Draw(image_copy)

# 定义参数
# center_pixel = (106, 140)  # 中心像素点的坐标 (i, j)
segment_length = 20  # 线段长度
orientation_step = 5  # 方向间隔（步长）
num_filters = 36  # 滤波器数量
flag = 0

for i in tqdm(range(0, image.size[0], 10)):
    for j in range(0, image.size[1], 10):
        if image_pixels[i, j] == 0:
            continue
        center_pixel = (i, j)
        # 计算每个方向上线段的平均灰度值
        average_gray_levels = []

        for angle in range(0, 180, orientation_step):
            # 计算线段的起始和结束点坐标
            start_point = (center_pixel[0], center_pixel[1] - int(segment_length / 2))
            end_point = (center_pixel[0], center_pixel[1] + int(segment_length / 2))

            # 创建线段
            rotated_image = image.rotate(angle, center=center_pixel)
            rotated_image_pixels = rotated_image.load()
            segment = []
            for i in range(max(0, start_point[1]), min(image.size[1]-1, end_point[1])):
                segment.append(rotated_image_pixels[center_pixel[0], i])
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
        line_endpoint_x = int(center_pixel[0] + line_length * np.sin(line_angle))
        line_endpoint_y = int(center_pixel[1] - line_length * np.cos(line_angle))
        start_point = center_pixel
        end_point = (line_endpoint_x, line_endpoint_y)
        # 绘制直线
        image_draw.line((start_point, end_point), fill=(0, 0, 256), width=1)

# 显示结果图像
image_copy.show()
# cv2.imshow("Image with Line and Center Point", image_draw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()