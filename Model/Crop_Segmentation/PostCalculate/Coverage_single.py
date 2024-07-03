import cv2
import tensorflow as tf
import yaml
import numpy as np

def mIoU(y_true, y_pred):
    true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
    pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
    cm = tf.math.confusion_matrix(true, pred, dtype=tf.float32)
    diag_item = tf.linalg.diag_part(cm)
    # return IoU_per
    mIoU = tf.reduce_mean(diag_item / (tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - tf.linalg.diag_part(cm)))

    # tf.summary.scalar('mean IoU', mIoU)


    return mIoU

with open('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']

image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Smart-Farm/image/0629_png_jpg.rf.35b395813db393b96f6679ca8f28cd15.jpg'
image_origin = tf.keras.preprocessing.image.load_img(image_path, target_size=(Height, Width))
image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)
image_origin = np.expand_dims(image_origin, axis=0)
image = image_origin / 255.0  # 标准化图像像素值


model = tf.keras.models.load_model('removeBackground.h5',custom_objects={"mIoU": mIoU})
probability_vector = model.predict(image)
max_prob_class_pot = np.argmax(probability_vector, axis=-1)
num_pixels_pot = np.sum(max_prob_class_pot == 1)


model = tf.keras.models.load_model('EfficientUnet3Plus7_200_epoch.h5')

probability_vector = model.predict(image)
# print(probability_vector.shape)

max_prob_class_leaf = np.argmax(probability_vector, axis=-1)


masked_leaf = max_prob_class_leaf * max_prob_class_pot

num_pixels_leaf = np.sum(masked_leaf == 1)
# print(num_pixels_leaf)

coverage = num_pixels_leaf / num_pixels_pot
print('coverage', coverage)


# 图片显示
import cv2 as cv
# 构建图像文件名（例如，image_0.png, image_1.png, ...）
# image_name = f"{i}.png"

'''
提取出叶子的部分，进行图像融合
'''
# 转换数组形状为 (224, 320)
arr = max_prob_class_leaf.squeeze()

# 创建形状为 (224, 320, 3) 的全零数组
green = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

# # 将值为 1 的像素映射为 (255, 255, 255)
green[arr == 1] = [0, 255, 0]

image = cv.imread(image_path)
image = cv.resize(image, (Width, Height))
max_prob_class_pot = np.squeeze(max_prob_class_pot)
max_prob_class_pot = cv2.merge([max_prob_class_pot, max_prob_class_pot, max_prob_class_pot])

masked_image = image * max_prob_class_pot

masked_image = np.array(masked_image, dtype=np.uint8)



# 读取原图和纱布图像
original_image = masked_image
green_overlay = green
# green_overlay[:, :, 1] = 255  # 将纱布图像的绿色通道设置为255

# 设置融合参数
alpha = 0.5  # 原图的权重
beta = 0.5  # 纱布图像的权重
gamma = 0  # 亮度调整

# 融合图像
blended_image = cv2.addWeighted(original_image, alpha, green_overlay, beta, gamma)

# 显示融合结果
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




combined_image = np.concatenate((image, masked_image), axis=1)
# 保存图像文件

cv.imshow('visualization', combined_image)
cv.waitKey()

