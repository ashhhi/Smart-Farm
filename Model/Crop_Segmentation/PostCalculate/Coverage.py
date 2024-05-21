import tensorflow as tf
import yaml
import numpy as np

with open('../config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']

image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm/augmented/bs1_0618_png_jpg.rf.aabb308df51c76b3fce4532e3212d661.jpg'
image_origin = tf.keras.preprocessing.image.load_img(image_path, target_size=(Height, Width))
image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)
image_origin = np.expand_dims(image_origin, axis=0)
image = image_origin / 255.0  # 标准化图像像素值

model = tf.keras.models.load_model('EfficientUnet3Plus7_200_epoch.h5')

probability_vector = model.predict(image)
# print(probability_vector.shape)

max_prob_class = np.argmax(probability_vector, axis=-1)

num_pixels_leaf = np.sum(max_prob_class == 1)
# print(num_pixels_leaf)

coverage = num_pixels_leaf / (Width*Height)
print('coverage', coverage)

