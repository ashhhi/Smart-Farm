from tensorflow.keras import layers, Model
import tensorflow as tf
from Model.Module.Convolution import ConvBlock, SeparableConvBlock

num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
Dbifpn = [3, 4, 5, 6, 7, 7, 8, 8]
# layers contain three different size feature maps from EfficientNet's last three different shape layers
def BiFPN(l, phi, attention=True):
    p3, p4, p5 = l
    p6 = layers.MaxPool2D((2, 2))(ConvBlock(p5, filters=num_filters[phi], kernel_size=1, ac=False))
    p7 = layers.MaxPool2D((2, 2))(p6)

    for i in range(Dbifpn[phi]):
        p3, p4, p5, p6, p7 = BiFPN_Block([p3, p4, p5, p6, p7], phi, attention)

    return p3, p4, p5, p6, p7


def BiFPN_Block(l, phi, attention):
    """ bifpn模块结构示意图
        P7_in -------------------------> P7_out -------->
           |-------------|                ↑
                         ↓                |
        P6_in ---------> P6_td ---------> P6_out -------->
           |-------------|--------------↑ ↑
                         ↓                |
        P5_in ---------> P5_td ---------> P5_out -------->
           |-------------|--------------↑ ↑
                         ↓                |
        P4_in ---------> P4_td ---------> P4_out -------->
           |-------------|--------------↑ ↑
                         |--------------↓ |
        P3_in -------------------------> P3_out -------->
    """
    p3, p4, p5, p6, p7 = l

    # adjust channel numbers to the same


    # Weighted feature fusion from original paper
    epsilon = 1e-4

    p4_w1 = layers.Activation('relu')(tf.Variable(tf.ones((2,), dtype=tf.float32), trainable=True))
    p4_w1_normalized = p4_w1 / (tf.reduce_sum(p4_w1, axis=0) + epsilon)

    p5_w1 = layers.Activation('relu')(tf.Variable(tf.ones((2,), dtype=tf.float32), trainable=True))
    p5_w1_normalized = p5_w1 / (tf.reduce_sum(p5_w1, axis=0) + epsilon)

    p6_w1 = layers.Activation('relu')(tf.Variable(tf.ones((2,), dtype=tf.float32), trainable=True))
    p6_w1_normalized = p6_w1 / (tf.reduce_sum(p6_w1, axis=0) + epsilon)

    p3_w2 = layers.Activation('relu')(tf.Variable(tf.ones((2,), dtype=tf.float32), trainable=True))
    p3_w2_normalized = p3_w2 / (tf.reduce_sum(p3_w2, axis=0) + epsilon)

    p4_w2 = layers.Activation('relu')(tf.Variable(tf.ones((3,), dtype=tf.float32), trainable=True))
    p4_w2_normalized = p4_w2 / (tf.reduce_sum(p4_w2, axis=0) + epsilon)

    p5_w2 = layers.Activation('relu')(tf.Variable(tf.ones((3,), dtype=tf.float32), trainable=True))
    p5_w2_normalized = p5_w2 / (tf.reduce_sum(p5_w2, axis=0) + epsilon)

    p6_w2 = layers.Activation('relu')(tf.Variable(tf.ones((3,), dtype=tf.float32), trainable=True))
    p6_w2_normalized = p6_w2 / (tf.reduce_sum(p6_w2, axis=0) + epsilon)

    p7_w2 = layers.Activation('relu')(tf.Variable(tf.ones((2,), dtype=tf.float32), trainable=True))
    p7_w2_normalized = p7_w2 / (tf.reduce_sum(p7_w2, axis=0) + epsilon)

    p3_in = ConvBlock(p3, filters=num_filters[phi], kernel_size=1, ac=False)
    p4_in = ConvBlock(p4, filters=num_filters[phi], kernel_size=1, ac=False)
    p5_in = ConvBlock(p5, filters=num_filters[phi], kernel_size=1, ac=False)
    p6_in = ConvBlock(p6, filters=num_filters[phi], kernel_size=1, ac=False)
    p7_in = ConvBlock(p7, filters=num_filters[phi], kernel_size=1, ac=False)

    if attention is False:
        p6_td = layers.Activation('swish')(p6_in + layers.UpSampling2D((2, 2), interpolation='bilinear')(p7_in))
        p6_td = SeparableConvBlock(p6_td, filters=num_filters[phi])

        p5_td = layers.Activation('swish')(p5_in + layers.UpSampling2D((2, 2), interpolation='bilinear')(p6_td))
        p5_td = SeparableConvBlock(p5_td, filters=num_filters[phi])

        p4_td = layers.Activation('swish')(p4_in + layers.UpSampling2D((2, 2), interpolation='bilinear')(p5_td))
        p4_td = SeparableConvBlock(p4_td, filters=num_filters[phi])

        p3_out = layers.Activation('swish')(p3_in + layers.UpSampling2D((2, 2), interpolation='bilinear')(p4_td))
        p3_out = SeparableConvBlock(p3_out, filters=num_filters[phi])

        p4_out = layers.Activation('swish')(p4_in + p4_td + layers.MaxPool2D((2, 2))(p3_out))
        p4_out = SeparableConvBlock(p4_out, filters=num_filters[phi])

        p5_out = layers.Activation('swish')(p5_in + p5_td + layers.MaxPool2D((2, 2))(p4_out))
        p5_out = SeparableConvBlock(p5_out, filters=num_filters[phi])

        p6_out = layers.Activation('swish')(p6_in + p6_td + layers.MaxPool2D((2, 2))(p5_out))
        p6_out = SeparableConvBlock(p6_out, filters=num_filters[phi])

        p7_out = layers.Activation('swish')(p7_in + layers.MaxPool2D((2, 2))(p6_out))
        p7_out = SeparableConvBlock(p7_out, filters=num_filters[phi])

    else:

        p6_td = layers.Activation('swish')(p6_w1_normalized[0] * p6_in + p6_w1_normalized[1] * layers.UpSampling2D((2, 2), interpolation='bilinear')(p7_in))
        p6_td = SeparableConvBlock(p6_td, filters=num_filters[phi])

        p5_td = layers.Activation('swish')(p5_w1_normalized[0] * p5_in + p5_w1_normalized[1] * layers.UpSampling2D((2, 2), interpolation='bilinear')(p6_td))
        p5_td = SeparableConvBlock(p5_td, filters=num_filters[phi])

        p4_td = layers.Activation('swish')(p4_w1_normalized[0] * p4_in + p4_w1_normalized[1] * layers.UpSampling2D((2, 2), interpolation='bilinear')(p5_td))
        p4_td = SeparableConvBlock(p4_td, filters=num_filters[phi])

        p3_out = layers.Activation('swish')(p3_w2_normalized[0] * p3_in + p3_w2_normalized[1] * layers.UpSampling2D((2, 2), interpolation='bilinear')(p4_td))
        p3_out = SeparableConvBlock(p3_out, filters=num_filters[phi])

        p4_out = layers.Activation('swish')(p4_w2_normalized[0] * p4_in + p4_w2_normalized[1] * p4_td + p4_w2_normalized[2] * layers.MaxPool2D((2, 2))(p3_out))
        p4_out = SeparableConvBlock(p4_out, filters=num_filters[phi])

        p5_out = layers.Activation('swish')(p5_w2_normalized[0] * p5_in + p5_w2_normalized[1] * p5_td + p5_w2_normalized[2] * layers.MaxPool2D((2, 2))(p4_out))
        p5_out = SeparableConvBlock(p5_out, filters=num_filters[phi])

        p6_out = layers.Activation('swish')(p6_w2_normalized[0] * p6_in + p6_w2_normalized[1] * p6_td + p6_w2_normalized[2] * layers.MaxPool2D((2, 2))(p5_out))
        p6_out = SeparableConvBlock(p6_out, filters=num_filters[phi])

        p7_out = layers.Activation('swish')(p7_w2_normalized[0] * p7_in + p7_w2_normalized[1] * layers.MaxPool2D((2, 2))(p6_out))
        p7_out = SeparableConvBlock(p7_out, filters=num_filters[phi])

    return [p3_out, p4_out, p5_out, p6_out, p7_out]