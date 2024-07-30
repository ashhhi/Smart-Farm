import numpy as np
from sklearn.model_selection import train_test_split
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
from loss import categorical_focal
from metrics import mIoU


os.chdir("./")
print(os.getcwd())

system = platform.system()
if system == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices('GPU')

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    Model_Used = yaml_data['Train']['Model_Used']
    epoch = yaml_data['Train']['Epoch']
    batch_size = yaml_data['Train']['Batch_Size']
    pre_trained_weights = yaml_data['Train']['Pre_Trained_Weights']
    class_map = yaml_data['Train']['Class_Map']
    image_path = yaml_data['Train']['image_path']
    label_path = yaml_data['Train']['label_path']


if Model_Used == 'Unet':
    from Model.Unet import Unet as create_model
elif Model_Used == 'EfficientUnet':
    from Model.EfficientUnet import efficientnet
    create_model = efficientnet(version=yaml_data['Models_Detail']['EfficientUnet3Plus']['version'])
elif Model_Used == 'EfficientUnet3Plus':
    if yaml_data['Models_Detail']['EfficientUnet3Plus']['layers'] == 5:
        from Model.EfficientUnet3Plus_5 import efficientnet
    else:
        from Model.EfficientUnet3Plus_7 import efficientnet
    create_model = efficientnet(version=yaml_data['Models_Detail']['EfficientUnet3Plus']['version'])
elif Model_Used == 'DeeplabV3':
    from Model.DeeplabV3 import DeeplabV3 as create_model
elif Model_Used == 'DeeplabV3Plus':
    from Model.DeeplabV3Plus import DeeplabV3Plus as create_model
elif Model_Used == 'SETR':
    from Model.SETR import vit_base_patch16_224_in21k as create_model
elif Model_Used == 'FCN':
    from Model.FCN import FCN as create_model
elif Model_Used == 'SegNet':
    from Model.SegNet import SegNet as create_model
elif Model_Used == 'RefineNet':
    from Model.RefineNet import RefineNet as create_model
else:
    from Model.Effi_Att_Unet3Plus_Det import efficientnet
    create_model = efficientnet()




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


def train():
    if pre_trained_weights:
        print('Load Pre Trained Weights:', pre_trained_weights)
        model = tf.keras.models.load_model(f"Model_save/again/{pre_trained_weights}", custom_objects={'mIoU': mIoU, 'categorical_focal_fixed': categorical_focal})
    else:
        print('Create new Model')
        model = create_model()
    model.summary()
    checkpoint_callback = ModelCheckpoint('Model_save/NewNet.h5', save_weights_only=False, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./History', write_graph=True)
    loss = False
    """
    尝试过的:
            [0.1, 1, 10]
            [0.5, 1, 3]
    """

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    if loss:
        model.compile(optimizer='Adam', loss=categorical_focal([0.1, 1, 10], 2), metrics=['accuracy', mIoU])
    elif len(class_map) >= 3:
        model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['accuracy', mIoU])
    else:
        model.compile(optimizer='Adam', loss="binary_crossentropy", metrics=['accuracy', mIoU])
    retval = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=1, batch_size=batch_size, callbacks=[checkpoint_callback, tensorboard_callback])
    with open('History/history.txt', 'w') as f:
        f.write(str(retval.history))
        print("Write History into History/history.txt")



if __name__ == '__main__':
    image_path, label_path, _ = Dataloader(image_path, label_path)
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