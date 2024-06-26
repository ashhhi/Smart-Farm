import tensorflow as tf
import yaml
import numpy as np
from tqdm import tqdm
import cv2 as cv

with open('../config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']

image_path = 'crystals.jpg'
image_origin = tf.keras.preprocessing.image.load_img(image_path, target_size=(Height, Width))
image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)
image_origin = np.expand_dims(image_origin, axis=0)
image = image_origin / 255.0  # 标准化图像像素值

model = tf.keras.models.load_model('EfficientUnet3Plus7_200_epoch.h5')

probability_vector = model.predict(image)

predicted_labels = np.argmax(probability_vector, axis=-1)
leaf_image = np.zeros_like(image)
crystals = np.zeros_like(image)

# 将mask为白色的区域显示在新图像上
leaf_image[predicted_labels > 0] = image[predicted_labels > 0]

threshold = 0.8 * np.ones_like(leaf_image)
crystals[np.all(leaf_image > threshold, axis=-1)] = leaf_image[np.all(leaf_image > threshold, axis=-1)]


# 显示结果
cv.imshow('Origin', image[0])
cv.imshow('leaf segmentation', leaf_image[0])
cv.imshow('crystals', crystals[0])
cv.waitKey(0)
cv.destroyAllWindows()
