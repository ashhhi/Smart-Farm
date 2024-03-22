import os
import cv2 as cv
import tensorflow as tf
import numpy as np

from Model.Crop_Segmentation.Model.EfficientUnet import efficientnet_b0 as create_model
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    output_dir = yaml_data['Path']['Predict_Save']
    pre_trained_weights = yaml_data['Train']['Pre_Trained_Weights']
    model_path = f"Model_save/{pre_trained_weights}"

    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

def preprocessing(image, label=False):
    image = cv.resize(image, (Width, Height))
    if label:
        red = np.array([0, 0, 255])
        green = np.array([0, 255, 0])
        one_hot = np.zeros_like(image)
        green_mask = np.all(image == green, axis=-1)
        red_mask = np.all(image == red, axis=-1)
        blue_mask = ~(red_mask | green_mask)
        one_hot[red_mask] = np.array([0, 0, 1])     # stem
        one_hot[green_mask] = np.array([0, 1, 0])   # leaf
        one_hot[blue_mask] = np.array([1, 0, 0])    # background
        return one_hot
    else:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image / 255.
        return image

model = tf.keras.models.load_model(model_path)


image_path = r'/Users/shijunshen/Documents/Code/dataset/Smart-Farm/Merge-1- 2- 4- 5- 5.1- 6-.v1i.voc/test/broccoli_3_day_7_11_png.rf.176918c840504d95cd466ffc551a85e5.jpg'


image = cv.imread(image_path)
image = preprocessing(image)

images = np.array([image])
probability_vector = model.predict(images)
color_map = {
    0: [255, 0, 0],    # Class 0: background
    1: [0, 255, 0],    # Class 1: leaf
    2: [0, 0, 255]     # Class 2: stem
}
predicted_labels = np.argmax(probability_vector, axis=-1)
colored_image = np.zeros((predicted_labels.shape[0], Height, Width, 3), dtype=np.uint8)
for n in range(predicted_labels.shape[0]):
    for i in range(Height):
        for j in range(Width):
            label = predicted_labels[n, i, j]
            colored_image[n, i, j] = color_map[label]

image = np.array(image)



# # 图像融合
# blended_image = cv.addWeighted(image * 255.0, 0.5, colored_image[0], 0.5, 0.0, dtype=cv.CV_8UC3)
# # 显示融合结果
# cv.imshow('Blended Image', blended_image)

# 图像显示
combined_image = np.concatenate((image, colored_image[0]), axis=1)
cv.imshow('prediction', combined_image)

cv.waitKey()