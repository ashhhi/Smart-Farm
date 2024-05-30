"""
refer to
https://blog.csdn.net/qq_37541097/article/details/121797301?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171567361516777224424859%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=171567361516777224424859&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-3-121797301-null-null.nonecase&utm_term=deeplab&spm=1018.2226.3001.4450
"""

import math
from typing import Union
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D, UpSampling2D
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation
import yaml
from Model.Crop_Segmentation.Model.Backbone import ResNet50


with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    Attention = yaml_data['Train']['Module']['Attention']
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

def CBA(inputs, filters, kernel_size=3, strides=1, padding='SAME'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Bottleneck2(inputs, filters, padding='same', expansion=4):
    identify = inputs

    x = Conv2D(filters / expansion, kernel_size=1, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters / expansion, kernel_size=3, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=1, padding=padding)(x)
    x = BatchNormalization()(x)

    x += identify
    x = Activation('relu')(x)
    return x



def ASPP(inputs):
    in_channel = inputs.shape[3]
    out_channel = in_channel // 8
    filter1 = tf.constant(value=1, shape=[1, 1, in_channel, out_channel], dtype=tf.float32)
    filter2 = tf.constant(value=1, shape=[3, 3, in_channel, out_channel], dtype=tf.float32)
    a1 = tf.nn.atrous_conv2d(inputs, filter1, rate=1, padding='SAME')
    a2 = tf.nn.atrous_conv2d(inputs, filter2, rate=12, padding='SAME')
    a3 = tf.nn.atrous_conv2d(inputs, filter2, rate=24, padding='SAME')
    a4 = tf.nn.atrous_conv2d(inputs, filter2, rate=36, padding='SAME')

    a5 = GlobalAvgPool2D()(inputs)
    size = inputs.shape[1:3]
    a5 = tf.reshape(a5, [-1, 1, 1, in_channel])
    a5 = Conv2D(out_channel, kernel_size=1, padding='SAME')(a5)
    a5 = BatchNormalization()(a5)
    a5 = Activation('relu')(a5)
    a5 = UpSampling2D(size=size, interpolation="bilinear")(a5)

    x = tf.keras.layers.concatenate([a1, a2, a3, a4, a5])
    x = Conv2D(filters=out_channel, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    y = Dropout(0.3)(x)

    return y

def DeeplabV3(input_shape=(Height, Width, 3)):
    img_input = layers.Input(shape=input_shape)

    """ ResNet-50 Backbone """
    x, _ = ResNet50.ResNet50(img_input)
    x = ASPP(inputs=x)

    """ Deeplab head """
    x = Conv2D(filters=256, kernel_size=3, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=3, kernel_size=1, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    """"""

    x = UpSampling2D(size=(8, 8), interpolation="bilinear")(x)
    x = Conv2D(filters=Class_Num, kernel_size=1, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model