import os
import random

import cv2 as cv
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from DataLoader.TestPoolDataloader import Dataloader
import yaml
from Model.SETR import ConcatClassTokenAddPosEmbed
from train import preprocessing

os.chdir("./")
print(os.getcwd())

with_label = False


custom_objects = {
    'ConcatClassTokenAddPosEmbed': ConcatClassTokenAddPosEmbed
}

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    output_dir = yaml_data['Predict']['save_path']
    is_all = yaml_data['Predict']['all']
    pre_trained_weights = yaml_data['Predict']['Pre_Trained_Weights']
    model_path = f"Model_save/Final/{pre_trained_weights}"
    class_map = yaml_data['Train']['Class_Map']
    image_path = yaml_data['Predict']['image_path']
    label_path = yaml_data['Predict']['label_path']

    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)


# model = create_model()
# model.load_weights(model_path)

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
model.compile()


image_path, label_path, image_name = Dataloader(image_path, label_path)
images = []
labels = []
print('Load Data and Preprocess...')
if is_all:
    num = len(image_path)
else:
    num = 10

for i in tqdm(range(num)):
    image = image_path[i]
    label = label_path[i]
    image = cv.imread(image)
    image = preprocessing(image)
    images.append(image)
    if with_label:
        label = cv.imread(label)
        label = preprocessing(label, True)
        labels.append(label)

images = np.array(images)
labels = np.array(labels) * 255
probability_vector = model.predict(images)
color_map = {}
cnt = 0
for item in class_map:
    color_map[str(cnt)] = list(reversed(class_map[item]))   # RGB --> BGR
    cnt += 1
predicted_labels = np.argmax(probability_vector, axis=-1)
colored_image = np.zeros((predicted_labels.shape[0], Height, Width, 3), dtype=np.uint8)
if with_label:
    ground_truth = np.argmax(labels, axis=-1)
    colored_label = np.zeros_like(colored_image, dtype=np.uint8)


for n in tqdm(range(predicted_labels.shape[0])):
    for i in range(Height):
        for j in range(Width):
            label = predicted_labels[n, i, j]

            colored_image[n, i, j] = color_map[str(label)]
            if with_label:
                gt = ground_truth[n, i, j]
                colored_label[n, i, j] = color_map[str(gt)]


#
# Save Together
# 遍历数组并保存每个元素为图像文件
for i, image in enumerate(colored_image):
    # 构建图像文件名（例如，image_0.png, image_1.png, ...）
    # image_name = f"{i}.png"
    name = image_name[i]

    if with_label:
        combined_image = np.concatenate((images[i] * 255, colored_label[i], image), axis=1)
    else:
        combined_image = np.concatenate((images[i] * 255, image), axis=1)
    # 保存图像文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = os.path.join(output_dir, name)

    cv.imwrite(image_path, combined_image)


# Save Independently
# for i, image in enumerate(colored_image):
#     gt_path = os.path.join(output_dir, 'gt')
#     pred_path = os.path.join(output_dir, 'pred')
#     if not os.path.exists(gt_path):
#         os.makedirs(gt_path)
#     if not os.path.exists(pred_path):
#         os.makedirs(pred_path)
#     # 构建图像文件名（例如，image_0.png, image_1.png, ...）
#     # name = f"{i}.png"
#     name = image_name[i]
#
#     # 保存图像文件
#     cv.imwrite(os.path.join(gt_path, name), labels[i])
#     cv.imwrite(os.path.join(pred_path, name), image)

