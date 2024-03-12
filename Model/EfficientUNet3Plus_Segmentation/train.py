import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from model_5 import efficientnet_b0 as create_model
import cv2 as cv
import tensorflow as tf
# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)
import yaml
import platform
from tqdm import tqdm
from DataLoader.TestPoolDataloader import Dataloader
system = platform.system()

if system == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices('GPU')

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
def preprocessing(image, label=False):
    image = cv.resize(image, (Width, Height))
    if label:
        red = np.array([0, 0, 255])
        green = np.array([0, 255, 0])
        one_hot = np.zeros_like(image)
        green_mask = np.all(image == green, axis=-1)
        red_mask = np.all(image == red, axis=-1)
        blue_mask = ~(red_mask | green_mask)
        one_hot[red_mask] = np.array([0, 0, 1])         # stem
        one_hot[green_mask] = np.array([0, 1, 0])       # leaf
        one_hot[blue_mask] = np.array([1, 0, 0])        # background
        return one_hot
    else:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image / 255.
        return image

def train():
    model = create_model()
    # model.load_weights("model_save/EfficientUnet_19.h5")
    model.summary()
    checkpoint_callback = ModelCheckpoint('model_save/EfficientUnet_{epoch:02d}.h5', save_weights_only=True, verbose=1)
    model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(images, labels, epochs=200, verbose=1, batch_size=8, callbacks=[checkpoint_callback])
    model.save('model_save/EfficientUnet_Final.h5')


if __name__ == '__main__':
    image_path, label_path = Dataloader()
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
    labels = np.array(labels)
    print('Finished and Start Training ...')
    train()