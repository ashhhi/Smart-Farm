import math
from typing import Union
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, UpSampling2D, Add
from Model.Module.Attention import attach_attention_module
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    Attention = yaml_data['Models_Detail']['Unet']['Attention']
    Class_Num = len(yaml_data['Train']['Class_Map'])

# 卷基层初始化方法
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

# 全连接层初始化方法
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same'):
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
    return conv





def FCN(input_shape=(Height, Width, 3),
                  dropout_rate=0.2,
                base_filter = 32,
                  activation="swish",
                  model_name="efficientnet"):
    img_input = layers.Input(shape=input_shape)
    # 下采样路径
    conv1 = conv_block(img_input, 96)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 256)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 384)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 384)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 256)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = conv_block(pool5, 4096)
    conv7 = conv_block(conv6, 4096)

    x = Conv2DTranspose(384, (3, 3), strides=(4, 4), padding='same')(conv7)
    pool4_ = Conv2DTranspose(384, (3, 3), strides=(2, 2), padding='same')(pool4)
    x = Add()([x, pool4_, pool3])
    x = UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
    x = conv_block(x, 3)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(Class_Num,
                      kernel_size=3,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softmax')(x)

    model = Model(img_input, x, name=model_name)

    return model
