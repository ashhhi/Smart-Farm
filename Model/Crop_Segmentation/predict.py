import os
import random
import time
import cv2 as cv
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from DataLoader.TestPoolDataloader import Dataloader
import yaml
from Model.SETR import ConcatClassTokenAddPosEmbed
from train import preprocessing
from evaluation import Evaluation

os.chdir("./")
print(os.getcwd())

custom_objects = {
    'ConcatClassTokenAddPosEmbed': ConcatClassTokenAddPosEmbed
}



with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']

    is_all = yaml_data['Predict']['all']
    pre_trained_weights = yaml_data['Predict']['Pre_Trained_Weights']
    model_path = f"{pre_trained_weights}"
    output_dir = os.path.join(yaml_data['Predict']['save_path'], model_path.split('/')[-1])
    class_map = yaml_data['Train']['Class_Map']
    image_path_ = yaml_data['Predict']['image_path']
    label_path_ = yaml_data['Predict']['label_path']

    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)


# model = create_model()
# model.load_weights(model_path)

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
model.compile()
model.summary()


image_path, label_path, image_name = Dataloader(image_path_, label_path_)

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
    if label_path_:
        label = cv.imread(label)
        label = preprocessing(label, True)
        labels.append(label)

# x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
# images = np.array(x_test)
# labels = np.array(y_test)

images = np.array(images)
labels = np.array(labels)

start_time = time.time()
probability_vector = model.predict(images)
end_time = time.time()
# 计算时间差
execution_time = end_time - start_time
# 打印运行时间
print("代码运行时间：", execution_time, "秒")
color_map = {}
cnt = 0
for item in class_map:
    color_map[str(cnt)] = list(reversed(class_map[item]))   # RGB --> BGR
    cnt += 1
predicted_labels = np.argmax(probability_vector, axis=-1)
colored_prediction = np.zeros((predicted_labels.shape[0], Height, Width, 3), dtype=np.uint8)
if label_path_:
    ground_truth = np.argmax(labels, axis=-1)
    colored_label = np.zeros_like(colored_prediction, dtype=np.uint8)

for n in tqdm(range(predicted_labels.shape[0])):
    for i in range(Height):
        for j in range(Width):
            label = predicted_labels[n, i, j]
            colored_prediction[n, i, j] = color_map[str(label)]
            if label_path_:
                gt = ground_truth[n, i, j]
                colored_label[n, i, j] = color_map[str(gt)]

print('start evaluation')
if label_path_:
    Evaluation(predicted_labels, ground_truth)
# Save Together
# 遍历数组并保存每个元素为图像文件
for i, image in enumerate(colored_prediction):
    # 构建图像文件名（例如，image_0.png, image_1.png, ...）
    # image_name = f"{i}.png"
    name = image_name[i]

    if label_path_:
        combined_image = np.concatenate((images[i] * 255, colored_label[i], image), axis=1)
    else:
        combined_image = np.concatenate((images[i] * 255, image), axis=1)
    # 保存图像文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = os.path.join(output_dir, name)

    cv.imwrite(image_path, combined_image)

if label_path_:
    # Save Independently
    for i, image in enumerate(colored_prediction):
        gt_path = os.path.join(output_dir, 'gt')
        pred_path = os.path.join(output_dir, 'pred')
        original_path = os.path.join(output_dir, 'origin')
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        if not os.path.exists(original_path):
            os.makedirs(original_path)
        # 构建图像文件名（例如，image_0.png, image_1.png, ...）
        name = f"{i}.png"

        # 保存图像文件
        cv.imwrite(os.path.join(gt_path, name), colored_label[i])
        cv.imwrite(os.path.join(pred_path, name), image)
        cv.imwrite(os.path.join(original_path, name), images[i]*255)