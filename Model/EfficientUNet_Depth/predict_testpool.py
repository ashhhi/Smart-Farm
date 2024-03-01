import os
import cv2 as cv
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# from Model.EfficientUNet_Depth.DataLoader.NYU_DataLoader import Dataloader
from Model.EfficientUNet_Depth.DataLoader.TestPoolDataloader import Dataloader
from model import efficientnet_b0 as create_model
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    model_path = yaml_data['Path']['Model']
    output_dir = yaml_data['Path']['Predict_Save']

    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

def preprocessing(image):
    image = cv.resize(image, (Width, Height))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.
    return image


model = create_model()
model.load_weights(model_path)


image_path, label_path = Dataloader()
images = []
labels = []
print('Load Data and Preprocess...')
for i in tqdm(range(len(image_path))):
    image = image_path[i]
    # label = label_path[i]
    image = preprocessing(cv.imread(image))
    # label = preprocessing(cv.imread(label, cv.IMREAD_GRAYSCALE))
    images.append(image)
    # labels.append(label)
images = np.array(images)
# labels = np.array(labels)
output = np.array(tf.squeeze(model.predict(images)*255, axis=-1)).astype(np.uint8)
# labels = labels * 255


print(output.shape)



# 遍历数组并保存每个元素为图像文件
for i, image in enumerate(output):
    # 构建图像文件名（例如，image_0.png, image_1.png, ...）
    image_name = f"image_{i}.png"

    # combined_image = np.concatenate((image, labels[i]), axis=1)

    gray_origin = cv.cvtColor(images[i]*255, cv.COLOR_BGR2GRAY)
    combined_image = np.concatenate((image, gray_origin), axis=1)
    # 保存图像文件
    image_path = os.path.join(output_dir, image_name)
    cv.imwrite(image_path, combined_image)

