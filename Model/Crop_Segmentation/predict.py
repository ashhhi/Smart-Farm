import os
import cv2 as cv
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from DataLoader.TestPoolDataloader import Dataloader
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    output_dir = yaml_data['Path']['Predict_Save']
    Model_Used = yaml_data['Train']['Model_Used']
    pre_trained_weights = yaml_data['Train']['Pre_Trained_Weights']
    model_path = f"Model_save/{pre_trained_weights}"

    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

if Model_Used == 'EfficientUnet3Plus_5':
    from Model.Crop_Segmentation.Model.EfficientUnet3Plus_5 import efficientnet_b0 as create_model
elif Model_Used == 'EfficientUnet3Plus_7':
    from Model.Crop_Segmentation.Model.EfficientUnet3Plus_7 import efficientnet_b0 as create_model
else:
    from Model.Crop_Segmentation.Model.EfficientUnet import efficientnet_b0 as create_model

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


image_path, label_path, image_name = Dataloader()
images = []
labels = []
print('Load Data and Preprocess...')
for i in tqdm(range(len(image_path))):
    image = image_path[i]
    label = label_path[i]
    image = cv.imread(image)
    image = preprocessing(image)
    label = cv.imread(label)
    label = preprocessing(label, True)
    images.append(image)
    labels.append(label)
images = np.array(images)
labels = np.array(labels) * 255
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


#
# Save Together
# 遍历数组并保存每个元素为图像文件
for i, image in enumerate(colored_image):
    # 构建图像文件名（例如，image_0.png, image_1.png, ...）
    # image_name = f"{i}.png"
    name = image_name[i]

    combined_image = np.concatenate((images[i] * 255, labels[i], image), axis=1)
    # 保存图像文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = os.path.join(output_dir, name)

    cv.imwrite(image_path, combined_image)


# Save Independently
for i, image in enumerate(colored_image):
    gt_path = os.path.join(output_dir, 'gt')
    pred_path = os.path.join(output_dir, 'pred')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    # 构建图像文件名（例如，image_0.png, image_1.png, ...）
    # name = f"{i}.png"
    name = image_name[i]

    # 保存图像文件
    cv.imwrite(os.path.join(gt_path, name), labels[i])
    cv.imwrite(os.path.join(pred_path, name), image)

