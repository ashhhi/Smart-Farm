from PIL import Image

# 打开图像文件
image = Image.open("/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/PostCalculate/photo/iphone.jpg")

# 将图像转换为RGB模式
image_rgb = image.convert("RGB")

# 获取图像的宽度和高度
width, height = image_rgb.size

# 创建一个新的空白图像
inverted_image = Image.new("RGB", (width, height))

# 遍历每个像素并取反颜色
for x in range(width):
    for y in range(height):
        r, g, b = image_rgb.getpixel((x, y))
        inverted_image.putpixel((x, y), (255 - r, 255 - g, 255 - b))

# 显示反转颜色后的图像
inverted_image.show()

# 保存反转颜色后的图像
inverted_image.save("inverted_image.jpg")