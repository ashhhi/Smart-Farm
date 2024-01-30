import tensorflow as tf

# 读取两张图像
image1 = tf.image.decode_image(tf.io.read_file('/Users/shijunshen/Documents/Code/dataset/NYUv2pics/nyu_depths/840.png'))
image2 = tf.image.decode_image(tf.io.read_file('/Users/shijunshen/Documents/Code/dataset/NYUv2pics/nyu_depths/810.png'))

# 将图像转为浮点类型，并归一化到[0, 1]范围
image1 = tf.image.convert_image_dtype(image1, tf.float32)
image2 = tf.image.convert_image_dtype(image2, tf.float32)

# 计算SSIM
ssim_value = tf.image.ssim(image1, image2, max_val=1.0)

# 打印SSIM值
print("SSIM:", ssim_value.numpy())