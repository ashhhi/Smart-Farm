"""
refer to
https://blog.csdn.net/qq_37541097/article/details/121797301?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171567361516777224424859%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=171567361516777224424859&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-3-121797301-null-null.nonecase&utm_term=deeplab&spm=1018.2226.3001.4450
https://blog.csdn.net/binlin199012/article/details/107155813
"""

import math
from typing import Union
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D, UpSampling2D, SeparableConv2D
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation
import yaml


with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    Attention = yaml_data['Train']['Module']['Attention']

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


def SepBlock(inputs, filters, kernel_size=3, padding='SAME', strides=1, dilation=1):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def CBA(inputs, filters, kernel_size=3, strides=1, padding='SAME'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Xception(inputs, output_stride=8):
    if output_stride == 8:  # os=8时，最后一次下采样dilation=1，之后dilation=4
        strides = (1, 1)
        dilations = (4, 4)
    elif output_stride == 16:  # os=16时，最后一次下采样dilation=1，之后dilation=2
        strides = (2, 1)
        dilations = (1, 2)
    elif output_stride == 32:  # os=32时，最后一次下采样dilation=1，之后dilation=1
        strides = (2, 2)
        dilations = (1, 1)

    """ Xception Backbone """
    # Entry flow
    x = CBA(inputs, filters=32, kernel_size=3, strides=2)
    x = CBA(x, filters=64, kernel_size=3)

    x_residue = Conv2D(filters=128, kernel_size=1, strides=2, padding='SAME')(x)
    x_residue = BatchNormalization()(x_residue)
    x = SepBlock(x, filters=128, kernel_size=3)
    x = SepBlock(x, filters=128, kernel_size=3)
    x = SepBlock(x, filters=128, kernel_size=3, strides=2)
    x = Activation('relu')(x + x_residue)
    low_level_feature = x

    x_residue = Conv2D(filters=256, kernel_size=1, strides=2, padding='SAME')(x)
    x_residue = BatchNormalization()(x_residue)
    x = SepBlock(x, filters=256, kernel_size=3)
    x = SepBlock(x, filters=256, kernel_size=3)
    x = SepBlock(x, filters=256, kernel_size=3, strides=2)
    x = Activation('relu')(x + x_residue)


    x_residue = Conv2D(filters=728, kernel_size=1, strides=strides[0], padding='SAME')(x)
    x_residue = BatchNormalization()(x_residue)
    x = SepBlock(x, filters=728, kernel_size=3, dilation=dilations[0])
    x = SepBlock(x, filters=728, kernel_size=3, dilation=dilations[0])
    x = SepBlock(x, filters=728, kernel_size=3, strides=strides[0], dilation=dilations[0])
    x = Activation('relu')(x + x_residue)

    # Middle Flow
    for i in range(16):
        x = SepBlock(x, filters=728, kernel_size=3, dilation=dilations[1])
        x = SepBlock(x, filters=728, kernel_size=3, dilation=dilations[1])
        x = SepBlock(x, filters=728, kernel_size=3, dilation=dilations[1])
        x = Activation('relu')(x + x_residue)

    # Exit Flow
    x_residue = Conv2D(filters=1024, kernel_size=1, strides=strides[1], padding='SAME')(x)
    x_residue = BatchNormalization()(x_residue)
    x = SepBlock(x, filters=728, kernel_size=3, dilation=dilations[1])
    x = SepBlock(x, filters=1024, kernel_size=3, dilation=dilations[1])
    x = SepBlock(x, filters=1024, kernel_size=3, strides=strides[1], dilation=dilations[1])
    x = Activation('relu')(x + x_residue)

    x = SepBlock(x, filters=1536, kernel_size=3)
    x = SepBlock(x, filters=1536, kernel_size=3)
    x = SepBlock(x, filters=2048, kernel_size=3)

    return x, low_level_feature