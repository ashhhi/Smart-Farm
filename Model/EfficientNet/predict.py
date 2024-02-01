import os
import cv2 as cv
import tensorflow as tf
import numpy as np



from model import efficientnet_b0 as create_model
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    model_path = yaml_data['Path']['Model']
    image_path = yaml_data['Path']['Datasets_Detail']

def preprocessing(image):
    image = cv.resize(image, (Width, Height))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.
    return image


model = create_model()
model.load_weights(model_path)
with open(image_path, 'r') as file:
    lines = file.readlines()
    images = []
    labels = []
    for item in lines:
        image_path = item.split()[0]
        label_path = item.split()[1]
        image = preprocessing(cv.imread(image_path))
        label = preprocessing(cv.imread(label_path, cv.IMREAD_GRAYSCALE))
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

output = np.array(tf.squeeze(model.predict(images)*255, axis=-1)).astype(np.uint8)
labels = labels * 255


print(output.shape)


# 创建保存图像的目录
output_dir = r"D:\dataset\NYU_DEPTH_V2\NYUv2pics\predict"
os.makedirs(output_dir, exist_ok=True)

# 遍历数组并保存每个元素为图像文件
for i, image in enumerate(output):
    # 构建图像文件名（例如，image_0.png, image_1.png, ...）
    image_name = f"image_{i}.png"

    combined_image = np.concatenate((image, labels[i]), axis=1)
    # 保存图像文件
    image_path = os.path.join(output_dir, image_name)
    cv.imwrite(image_path, combined_image)
