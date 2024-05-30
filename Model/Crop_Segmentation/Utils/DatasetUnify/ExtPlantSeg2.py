import os
import shutil

import yaml
from PIL import Image

plant_segmentation_path = ['/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Tray/Ara2012', '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Tray/Ara2013-Canon']
leaf_segmentation_path = ['/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Plant/Ara2012', '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon', '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Plant/Tobacco']


saved_segmentation_image = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Segmentation/image'
saved_segmentation_mask = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Segmentation/mask'
saved_segmentation_new_mask = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg2/Plant_Phenotyping_Datasets/Segmentation/new_mask'

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    class_map = yaml_data['Train']['Class_Map']
# 遍历源文件夹中的所有文件
for p in plant_segmentation_path:
    for filename in os.listdir(p):
        # 检查文件名是否包含 'fg'
        if 'fg' in filename:
            # 拼接源文件路径
            source_file_path = os.path.join(p, filename)

            # 拼接目标文件路径
            target_file_path = os.path.join(saved_segmentation_mask, filename.replace('fg', 'rgb'))

            # 复制文件
            shutil.copy2(source_file_path, target_file_path)

            print(f"File copied: {filename}")
        elif 'rgb' in filename:
            # 拼接源文件路径
            source_file_path = os.path.join(p, filename)

            # 拼接目标文件路径
            target_file_path = os.path.join(saved_segmentation_image, filename)

            # 复制文件
            shutil.copy2(source_file_path, target_file_path)

            print(f"File copied: {filename}")

# 遍历源文件夹中的所有文件
for p in leaf_segmentation_path:
    for filename in os.listdir(p):
        # 检查文件名是否包含 'label'
        if 'label' in filename:
            # 拼接源文件路径
            source_file_path = os.path.join(p, filename)

            # 拼接目标文件路径
            target_file_path = os.path.join(saved_segmentation_mask, filename.replace('label', 'rgb'))

            # 复制文件
            shutil.copy2(source_file_path, target_file_path)

            print(f"File copied: {filename}")
        elif 'rgb' in filename:
            # 拼接源文件路径
            source_file_path = os.path.join(p, filename)

            # 拼接目标文件路径
            target_file_path = os.path.join(saved_segmentation_image, filename)

            # 复制文件
            shutil.copy2(source_file_path, target_file_path)

            print(f"File copied: {filename}")

# 遍历源文件夹中的所有文件
for filename in os.listdir(saved_segmentation_mask):
    # 检查文件是否为图片文件
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.gif'):
        # 拼接源文件路径
        source_file_path = os.path.join(saved_segmentation_mask, filename)

        # 拼接目标文件路径
        target_file_path = os.path.join(saved_segmentation_new_mask, filename)

        # 打开图片文件
        image = Image.open(source_file_path)

        # 将 RGB 图像转换为灰度图像
        grayscale_image = image.convert('L')

        # 创建一个空的图像,用于存储修改后的图像
        modified_image = Image.new('RGB', grayscale_image.size)

        # 遍历灰度图像的每个像素
        for x in range(grayscale_image.width):
            for y in range(grayscale_image.height):
                # 获取当前像素的灰度值
                gray_value = grayscale_image.getpixel((x, y))

                Leaf_color = class_map['Leaf']
                if gray_value != 0:
                    modified_image.putpixel((x, y), tuple(Leaf_color))


        # 保存修改后的图像
        modified_image.save(target_file_path)

        print(f"Image processed: {filename}")