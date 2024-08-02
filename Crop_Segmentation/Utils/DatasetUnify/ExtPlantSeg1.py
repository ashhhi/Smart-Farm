import os

import yaml
from PIL import Image
import numpy as np

image_path = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/images'
origin_label_path = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/masks'
new_mask_label = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/new_masks'
with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    class_map = yaml_data['Train']['Class_Map']

# 遍历输入文件夹下的所有图片
for filename in os.listdir(origin_label_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 读取图片
        img_path = os.path.join(origin_label_path, filename)
        img = Image.open(img_path)

        # 将非黑色部分变为纯绿色

        Leaf_color = class_map['Leaf']
        img_array = np.array(img)
        black_pixels = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)

        img_array[~black_pixels, 0] = Leaf_color[0]  # 将蓝色通道设为 0
        img_array[~black_pixels, 1] = Leaf_color[1]  # 将绿色通道设为 255
        img_array[~black_pixels, 2] = Leaf_color[2]  # 将红色通道设为 0

        # 创建新的 PIL 图像对象并保存
        new_img = Image.fromarray(img_array)
        new_img_path = os.path.join(new_mask_label, filename)
        new_img.save(new_img_path)

print("Processing completed. New images saved to", new_mask_label)


# 遍历源文件夹中的所有文件
for filename in os.listdir(image_path):
    # 检查文件是否为图片文件
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.gif'):
        # 拼接源文件路径
        source_file_path = os.path.join(image_path, filename)

        # 创建新的文件名 (例如在原文件名前加上 'new_')
        new_filename = str(filename.split('.')[0]) + '.jpg'

        # 拼接新的文件路径
        new_file_path = os.path.join(image_path, new_filename)

        # 重命名文件
        os.rename(source_file_path, new_file_path)

        print(f"File renamed: {filename} -> {new_filename}")

# 遍历源文件夹中的所有文件
for filename in os.listdir(new_mask_label):
    # 检查文件是否为图片文件
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.gif'):
        # 拼接源文件路径
        source_file_path = os.path.join(new_mask_label, filename)

        # 创建新的文件名 (例如在原文件名前加上 'new_')
        new_filename = str(filename.split('.')[0]) + '.png'

        # 拼接新的文件路径
        new_file_path = os.path.join(new_mask_label, new_filename)

        # 重命名文件
        os.rename(source_file_path, new_file_path)

        print(f"File renamed: {filename} -> {new_filename}")