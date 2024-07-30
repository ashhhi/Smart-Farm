import os
import random

import cv2 as cv
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import yaml

os.chdir("./")
print(os.getcwd())

def mIoU(y_true, y_pred):
    true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
    pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
    cm = tf.math.confusion_matrix(true, pred, dtype=tf.float32)
    diag_item = tf.linalg.diag_part(cm)
    # return IoU_per
    mIoU = tf.reduce_mean(diag_item / (tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - tf.linalg.diag_part(cm)))

    # tf.summary.scalar('mean IoU', mIoU)


    return mIoU
def preprocessing(image, label=False):
    image = cv.resize(image, (Width, Height))
    one_hot = np.zeros((Height, Width, len(class_map)))
    if label:
        cnt = 0
        for item in class_map:
            temp = np.array(list(reversed(class_map[item])))
            mask = np.all(image == temp, axis=-1)
            oh = len(class_map) * [0]
            oh[cnt] = 1
            one_hot[mask] = np.array(oh)
            cnt += 1
        mask = (one_hot == [0]*len(class_map)).all(axis=2)
        one_hot[mask] = [1] + [0] * (len(class_map)-1)
        return one_hot
    else:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image / 255.
        return image


def Dataloader(image_path, label_path):
    g = os.walk(image_path)
    image = []
    label = []
    image_name = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.split('.')[-1] == 'jpg':
                image.append(str(os.path.join(path, file_name)))
                label.append(str(os.path.join(label_path, file_name.replace('.jpg', '.png'))))
                image_name.append(file_name)

    # 处理xml文件
    return image, label, image_name


with open('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    output_dir = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Smart-Farm/coverage_pred'
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

model = tf.keras.models.load_model('removeBackground.h5', custom_objects={"mIoU": mIoU}, compile=False)
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
    if label_path:
        label = cv.imread(label)
        label = preprocessing(label, True)
        labels.append(label)

images = np.array(images)
labels = np.array(labels) * 255

model = tf.keras.models.load_model('removeBackground.h5', custom_objects={"mIoU": mIoU})
probability_vector = model.predict(images)
max_prob_class_pot = np.argmax(probability_vector, axis=-1)

num_pixels_pot_list = []
for item in max_prob_class_pot:
    num_pixels_pot = np.sum(item == 1)
    num_pixels_pot_list.append(num_pixels_pot)

model = tf.keras.models.load_model('EfficientUnet3Plus.h5', custom_objects={"mIoU": mIoU})
probability_vector = model.predict(images)
# print(probability_vector.shape)
max_prob_class_leaf = np.argmax(probability_vector, axis=-1)
masked_leaf = max_prob_class_leaf * max_prob_class_pot

num_pixels_leaf_list = []
for item in max_prob_class_leaf:
    num_pixels_leaf = np.sum(item == 1)
    num_pixels_leaf_list.append(num_pixels_leaf)

# print(num_pixels_leaf)
coverage = [x / y for x, y in zip(num_pixels_leaf_list, num_pixels_pot_list)]
print('coverage', coverage)

'''
提取出叶子的部分，进行图像融合
'''

print('write')
for i in tqdm(range(len(max_prob_class_leaf))):
    # 转换数组形状为 (224, 320)
    arr = max_prob_class_leaf[i].squeeze()
    # 创建形状为 (224, 320, 3) 的全零数组
    green = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    # # 将值为 1 的像素映射为 (255, 255, 255)
    green[arr == 1] = [0, 255, 0]

    max_prob_class_pot_ = np.squeeze(max_prob_class_pot[i])
    max_prob_class_pot_ = cv.merge([max_prob_class_pot_, max_prob_class_pot_, max_prob_class_pot_])
    masked_image = images[i].squeeze() * max_prob_class_pot_ * 255
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
    blended_image = cv.addWeighted(original_image, alpha, green_overlay, beta, gamma)

    combined_image = np.concatenate((images[i] * 255, blended_image), axis=1)
    # 保存图像文件

    print(os.path.join(output_dir, f'{round(coverage[i], 3)}.jpg'))
    cv.imwrite(os.path.join(output_dir, f'{round(coverage[i], 3)}.jpg'), combined_image)
