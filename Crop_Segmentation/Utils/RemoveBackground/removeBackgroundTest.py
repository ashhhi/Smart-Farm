import os
from PIL import Image

# 原图路径
original_image_path = '/Users/shijunshen/Documents/Code/dataset/RemoveBackground/train'
# 黑白图路径
bw_image_path = '/Users/shijunshen/Documents/Code/dataset/RemoveBackground/mask'

# 输出图片路径
output_path = '/Users/shijunshen/Documents/Code/dataset/RemoveBackground/removedImage'

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)

# 遍历原图和黑白图路径下的图片
for filename in os.listdir(original_image_path):
    if filename.split('.')[-1] == 'xml':
        continue
    # 构建原图和黑白图的完整路径
    try:
        original_image = Image.open(os.path.join(original_image_path, filename))
        bw_image = Image.open(os.path.join(bw_image_path, '.'.join(filename.split('.')[:-1])+'.png'))
    except Exception as e:
        print(e)
        continue

    # 创建一个与原图大小相同的新图片
    output_image = Image.new('RGB', original_image.size)

    # 遍历原图每个像素,如果对应的黑白图像素是白色,则拷贝到输出图
    for x in range(original_image.width):
        for y in range(original_image.height):
            if bw_image.getpixel((x, y)) == (255, 255, 255):
                output_image.putpixel((x, y), original_image.getpixel((x, y)))

    # 保存输出图片
    output_image.save(os.path.join(output_path, filename))