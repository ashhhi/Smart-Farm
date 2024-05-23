import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D, UpSampling2D
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation
import yaml

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