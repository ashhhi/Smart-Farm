import os

image_path = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/images'
mask_path = '/Users/shijunshen/Documents/Code/dataset/ExtPlantSeg1/new_masks'

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
for filename in os.listdir(mask_path):
    # 检查文件是否为图片文件
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.gif'):
        # 拼接源文件路径
        source_file_path = os.path.join(mask_path, filename)

        # 创建新的文件名 (例如在原文件名前加上 'new_')
        new_filename = str(filename.split('.')[0]) + '.png'

        # 拼接新的文件路径
        new_file_path = os.path.join(mask_path, new_filename)

        # 重命名文件
        os.rename(source_file_path, new_file_path)

        print(f"File renamed: {filename} -> {new_filename}")