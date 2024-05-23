import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import cv2 as cv
import tensorflow as tf
# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)
import yaml
import platform
from tqdm import tqdm
from DataLoader.TestPoolDataloader import Dataloader

os.chdir("./")
print(os.getcwd())

system = platform.system()
if system == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices('GPU')

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    Model_Used = yaml_data['Train']['Model_Used']
    epoch = yaml_data['Train']['Epoch']
    batch_size = yaml_data['Train']['Batch_Size']
    pre_trained_weights = yaml_data['Train']['Pre_Trained_Weights']


if Model_Used == 'Unet':
    from Model.Unet import Unet as create_model
elif Model_Used == 'EfficientUnet3':
    from Model.EfficientUnet import efficientnet_b0 as create_model
elif Model_Used == 'EfficientUnet3Plus':
    if yaml_data['Models_Detail']['EfficientUnet3Plus']['layers'] == 5:
        from Model.EfficientUnet3Plus_5 import efficientnet_b0 as create_model
    else:
        from Model.EfficientUnet3Plus_7 import efficientnet_b0 as create_model
elif Model_Used == 'DeeplabV3':
    from Model.DeeplabV3 import DeeplabV3 as create_model
elif Model_Used == 'DeeplabV3Plus':
    from Model.DeeplabV3Plus import DeeplabV3Plus as create_model
elif Model_Used == 'SETR':
    from Model.SETR import vit_base_patch16_224_in21k as create_model
else:
    assert 0, 'No model chosed!'

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
    if pre_trained_weights:
        print('Load Pre Trained Weights:', pre_trained_weights)
        model = tf.keras.models.load_model(f"Model_save/{pre_trained_weights}")
    else:
        print('Create new Model')
        model = create_model()
    model.summary()
    checkpoint_callback = ModelCheckpoint('Model_save/NewNet_{epoch:02d}.h5', save_weights_only=False, verbose=1)
    model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['accuracy'])
    retval = model.fit(images, labels, epochs=epoch, verbose=1, batch_size=batch_size, callbacks=[checkpoint_callback])
    with open('History/history.txt', 'w') as f:
        f.write(str(retval.history))
        print("Write History into History/history.txt")



if __name__ == '__main__':
    image_path, label_path, _ = Dataloader()
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
    print('Model:', Model_Used)
    train()