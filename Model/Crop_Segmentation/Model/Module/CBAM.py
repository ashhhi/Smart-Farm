import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D, UpSampling2D, BatchNormalization,DepthwiseConv2D

def Channal(input, out_dim, ratio=16):
    squeeze1 = GlobalAvgPool2D()(input)
    excitation1 = Dense(units=out_dim / ratio)(squeeze1)
    excitation1 = Activation('relu')(excitation1)
    excitation1 = Dense(units=out_dim)(excitation1)
    excitation1 = Activation('sigmoid')(excitation1)
    excitation1 = tf.reshape(excitation1, [-1, 1, 1, out_dim])

    squeeze2 = GlobalMaxPool2D()(input)
    excitation2 = Dense(units=out_dim / ratio)(squeeze2)
    excitation2 = Activation('relu')(excitation2)
    excitation2 = Dense(units=out_dim)(excitation2)
    excitation2 = Activation('sigmoid')(excitation2)
    excitation2 = tf.reshape(excitation2, [-1, 1, 1, out_dim])

    excitation = excitation1 + excitation2
    scale = input * excitation
    return scale


def Spatial(input):
    x1 = tf.reduce_max(input, 3)
    x2 = tf.reduce_mean(input, 3)
    x1 = tf.reshape(x1, (tf.shape(x1)[0], tf.shape(x1)[1], tf.shape(x1)[2], 1))
    x2 = tf.reshape(x2, (tf.shape(x2)[0], tf.shape(x2)[1], tf.shape(x2)[2], 1))
    x = tf.concat((x1, x2), axis=3)
    x = Conv2D(kernel_size=7, filters=1, strides=1, padding='same')(x)
    x = Activation('sigmoid')(x)
    y = x * input
    return y


def CBAM(input, initial_filters):
    x = Channal(input, initial_filters)
    x = Spatial(x)
    return x