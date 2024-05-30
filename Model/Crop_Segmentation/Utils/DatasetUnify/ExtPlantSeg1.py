import os
from PIL import Image
import numpy as np

origin_label_path = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/masks'
new_mask_label = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/new_masks'

# 遍历输入文件夹下的所有图片
for filename in os.listdir(origin_label_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 读取图片
        img_path = os.path.join(origin_label_path, filename)
        img = Image.open(img_path)

        # 将黑色部分变为蓝色
        img_array = np.array(img)
        black_pixels = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)
        img_array[black_pixels, 0] = 0  # 将蓝色通道设为 0
        img_array[black_pixels, 1] = 0  # 将绿色通道设为 0
        img_array[black_pixels, 2] = 255  # 将红色通道设为 255

        img_array[~black_pixels, 0] = 0  # 将蓝色通道设为 0
        img_array[~black_pixels, 1] = 255  # 将绿色通道设为 255
        img_array[~black_pixels, 2] = 0  # 将红色通道设为 0

        # 创建新的 PIL 图像对象并保存
        new_img = Image.fromarray(img_array)
        new_img_path = os.path.join(new_mask_label, filename)
        new_img.save(new_img_path)

print("Processing completed. New images saved to", new_mask_label)