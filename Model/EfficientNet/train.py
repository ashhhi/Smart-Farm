import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

from Model.EfficientNet.evaluate import mean_relative_error
from Model.EfficientNet.loss import loss2
from model import efficientnet_b0 as create_model
import cv2 as cv
import tensorflow as tf
# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)
import pdb
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    image_path = yaml_data['Path']['Datasets_Detail']
def preprocessing(image):
    image = cv.resize(image, (Width, Height))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.
    return image


def train():
    model = create_model()
    # model.load_weights("model_save/EfficientUnet_19.h5")
    model.summary()
    checkpoint_callback = ModelCheckpoint('model_save/EfficientUnet_{epoch:02d}.h5', save_weights_only=True, verbose=1)
    model.compile(optimizer='Adam', loss=loss2, metrics=['mse', 'mae', mean_relative_error])
    model.fit(images, labels, epochs=200, verbose=1, batch_size=8, callbacks=[checkpoint_callback])
    model.save('model_save/EfficientUnet_Final.h5')


if __name__ == '__main__':
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

    train()