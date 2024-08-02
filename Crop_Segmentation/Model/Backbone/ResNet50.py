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


with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
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

def Bottleneck1(inputs, filters, strides=1, expansion=4):
    identify = inputs
    identify = Conv2D(filters * expansion, strides=strides, kernel_size=1, padding='SAME')(identify)
    identify = BatchNormalization()(identify)

    x = CBA(inputs, filters=filters, strides=strides, kernel_size=1)
    x = CBA(x, filters=filters, strides=1, kernel_size=3)

    x = Conv2D(filters * expansion, kernel_size=1, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x+identify)

    return x

def Bottleneck2(inputs, filters, expansion=4):
    identify = inputs

    x = CBA(inputs, filters=filters//expansion, kernel_size=1)
    x = CBA(x, filters=filters // expansion, kernel_size=3)

    x = Conv2D(filters, kernel_size=1, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x+identify)
    return x


def ResNet50(inputs):
    """ ResNet-50 Backbone """
    x = CBA(inputs, filters=64, kernel_size=7, strides=2)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Stage 1
    x = Bottleneck1(inputs=x, filters=64)
    for i in range(2):
        x = Bottleneck2(inputs=x, filters=256)

    low_level_feature = x

    # Stage 2
    x = Bottleneck1(inputs=x, strides=2, filters=128)
    for i in range(2):
        x = Bottleneck2(inputs=x, filters=512)

    # Stage 3
    x = Bottleneck1(inputs=x, strides=2, filters=256)
    for i in range(5):
        x = Bottleneck2(inputs=x, filters=1024)


    # Stage 4
    x = Bottleneck1(inputs=x, filters=512)
    for i in range(2):
        x = Bottleneck2(inputs=x, filters=2048)

    return x, low_level_feature