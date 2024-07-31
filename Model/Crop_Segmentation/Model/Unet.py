import math
from typing import Union
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
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



def UnetDecoder(input, filters, stride, concat, dropout_rate):
    x = layers.Conv2DTranspose(filters, (3, 3), strides=(stride, stride), padding='same')(input)
    x = layers.concatenate([concat, x])
    x = layers.Conv2D(filters,
                      kernel_size=3,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)

    if Attention:
        x = attach_attention_module(x, Attention)
    # x = layers.Activation("swish")(x)
    return x


def Unet(input_shape=(Height, Width, 3),
                  dropout_rate=0.2,
                base_filter = 16,
                  activation="swish",
                  model_name="efficientnet"):
    img_input = layers.Input(shape=input_shape)
    # 下采样路径
    conv1 = conv_block(img_input, base_filter)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, base_filter * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, base_filter * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, base_filter * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, base_filter * 16)

    x = UnetDecoder(conv5, base_filter * 8, 2, conv4, dropout_rate)
    x = UnetDecoder(x, base_filter * 4, 2, conv3, dropout_rate)
    x = UnetDecoder(x, base_filter * 2, 2, conv2, dropout_rate)
    x = UnetDecoder(x, base_filter, 2, conv1, dropout_rate)
    # x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(Class_Num,
                      kernel_size=3,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softmax')(x)

    model = Model(img_input, x, name=model_name)

    return model
