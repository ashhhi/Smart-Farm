import math
from typing import Union
from tensorflow.keras import layers, Model
from Model.Module.Attention import attach_attention_module
import yaml
import tensorflow as tf


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


# 当stride=2的时候是如何对特征矩阵进行padding填充的
def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments:
      input_size: Input tensor size.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    kernel_size = (kernel_size, kernel_size)

    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


# MBConv模块
def block(inputs,
          activation: str = "swish",
          drop_rate: float = 0.,
          name: str = "",
          input_channel: int = 32,
          output_channel: int = 16,
          kernel_size: int = 3,
          strides: int = 1,
          expand_ratio: int = 1,
          use_se: bool = True,
          se_ratio: float = 0.25):
    """An inverted residual block.

      Arguments:
          inputs: input tensor.
          activation: activation function.
          drop_rate: float between 0 and 1, fraction of the input units to drop.
          name: string, block label.
          input_channel: integer, the number of input filters.
          output_channel: integer, the number of output filters.
          kernel_size: integer, the dimension of the convolution window.
          strides: integer, the stride of the convolution.
          expand_ratio: integer, scaling coefficient for the input filters.
          use_se: whether to use se
          se_ratio: float between 0 and 1, fraction to squeeze the input filters.

      Returns:
          output tensor for the block.
      """
    # Expansion phase
    filters = input_channel * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters=filters,
                          kernel_size=1,
                          padding="same",
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + "expand_conv")(inputs)
        x = layers.BatchNormalization(name=name + "expand_bn")(x)
        x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(filters, kernel_size),
                                 name=name + "dwconv_pad")(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=strides,
                               padding="same" if strides == 1 else "valid",
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + "dwconv")(x)
    x = layers.BatchNormalization(name=name + "bn")(x)
    x = layers.Activation(activation, name=name + "activation")(x)

    if use_se:
        filters_se = int(input_channel * se_ratio)
        se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        se = layers.Reshape((1, 1, filters), name=name + "se_reshape")(se)
        se = layers.Conv2D(filters=filters_se,
                           kernel_size=1,
                           padding="same",
                           activation=activation,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + "se_reduce")(se)
        se = layers.Conv2D(filters=filters,
                           kernel_size=1,
                           padding="same",
                           activation="sigmoid",
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + "se_expand")(se)
        x = layers.multiply([x, se], name=name + "se_excite")

    # Output phase
    x = layers.Conv2D(filters=output_channel,
                      kernel_size=1,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + "project_conv")(x)
    x = layers.BatchNormalization(name=name + "project_bn")(x)
    if strides == 1 and input_channel == output_channel:
        if drop_rate > 0:
            x = layers.Dropout(rate=drop_rate,
                               noise_shape=(None, 1, 1, 1),  # binary dropout mask
                               name=name + "drop")(x)
        x = layers.add([x, inputs], name=name + "add")

    return x




def efficient_net(width_coefficient,
                  depth_coefficient,
                  input_shape=(Height, Width, 3),
                  dropout_rate=0.2,
                  drop_connect_rate=0.2,
                  activation="swish",
                  model_name="efficientnet"):
    """Instantiates the EfficientUNet_Depth architecture using given scaling coefficients.

      Reference:
      - [EfficientUNet_Depth: Rethinking Model Scaling for Convolutional Neural Networks](
          https://arxiv.org/abs/1905.11946) (ICML 2019)

      Optionally loads weights pre-trained on ImageNet.
      Note that the data format convention used by the model is
      the one specified in your Keras config at `~/.keras/keras.json`.

      Arguments:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        input_shape: tuple, default input image shape(not including the batch size).
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        activation: activation function.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

      Returns:
        A `keras.Model` instance.
    """

    # kernel_size, repeats, in_channel, out_channel, exp_ratio, strides, SE
    block_args = [[3, 1, 32, 16, 1, 1, True],
                  [3, 2, 16, 24, 6, 2, True],
                  [5, 2, 24, 40, 6, 2, True],
                  [3, 3, 40, 80, 6, 2, True],
                  [5, 3, 80, 112, 6, 1, True],
                  [5, 4, 112, 192, 6, 2, True],
                  [3, 1, 192, 320, 6, 1, True]]

    def round_filters(filters, divisor=8):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    img_input = layers.Input(shape=input_shape)

    # data preprocessing
    # x = layers.experimental.preprocessing.Rescaling(1. / 255.)(img_input)
    # x = layers.experimental.preprocessing.Normalization()(x)

    # first conv2d (224,224,3) -> (112,112,32)
    x = layers.ZeroPadding2D(padding=correct_pad(input_shape[:2], 3),
                             name="stem_conv_pad")(img_input)
    x = layers.Conv2D(filters=round_filters(32),
                      kernel_size=3,
                      strides=2,
                      padding="valid",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # build blocks
    b = 0
    Concatenate_waiting = []
    num_blocks = float(sum(round_repeats(i[1]) for i in block_args))
    for i, args in enumerate(block_args):
        assert args[1] > 0
        # Update block input and output filters based on depth multiplier.
        args[2] = round_filters(args[2])  # input_channel
        args[3] = round_filters(args[3])  # output_channel

        for j in range(round_repeats(args[1])):
            x = block(x,
                      activation=activation,
                      drop_rate=drop_connect_rate * b / num_blocks,
                      name="block{}{}_".format(i + 1, chr(j + 97)),
                      kernel_size=args[0],
                      input_channel=args[2] if j == 0 else args[3],
                      output_channel=args[3],
                      expand_ratio=args[4],
                      strides=args[5] if j == 0 else 1,
                      use_se=args[6])
            b += 1
        Concatenate_waiting.append(x)

    # Unet3+ Architecture
    base_channel = Concatenate_waiting[0].shape[-1]
    if Attention:
        tmp = []
        for layer in Concatenate_waiting:
            tmp.append(attach_attention_module(layer, Attention))
        Concatenate_waiting = tmp

    e1_1 = Concatenate_waiting[0]
    e1_2 = tf.keras.layers.MaxPooling2D((2, 2))(Concatenate_waiting[0])
    e1_3 = tf.keras.layers.MaxPooling2D((4, 4))(Concatenate_waiting[0])
    e1_4 = tf.keras.layers.MaxPooling2D((8, 8))(Concatenate_waiting[0])
    e1_5 = e1_4
    e1_6 = tf.keras.layers.MaxPooling2D((16, 16))(Concatenate_waiting[0])

    e2_2 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(Concatenate_waiting[1])
    e2_3 = tf.keras.layers.MaxPooling2D((2, 2))(Concatenate_waiting[1])
    e2_3 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e2_3)
    e2_4 = tf.keras.layers.MaxPooling2D((4, 4))(Concatenate_waiting[1])
    e2_4 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e2_4)
    e2_5 = e2_4
    e2_6 = tf.keras.layers.MaxPooling2D((8, 8))(Concatenate_waiting[1])
    e2_6 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e2_6)

    e3_3 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(Concatenate_waiting[2])
    e3_4 = tf.keras.layers.MaxPooling2D((2, 2))(Concatenate_waiting[2])
    e3_4 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e3_4)
    e3_5 = e3_4
    e3_6 = tf.keras.layers.MaxPooling2D((4, 4))(Concatenate_waiting[2])
    e3_6 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e3_6)

    e4_4 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(Concatenate_waiting[3])
    e4_5 = e4_4
    e4_6 = tf.keras.layers.MaxPooling2D((2, 2))(Concatenate_waiting[3])
    e4_6 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e4_6)

    e5_5 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(Concatenate_waiting[4])
    e5_6 = tf.keras.layers.MaxPooling2D((2, 2))(Concatenate_waiting[4])
    e5_6 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(e5_6)

    e6_6 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                      kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(Concatenate_waiting[5])

    d7_1 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(16, 16), padding='same')(
        Concatenate_waiting[6])
    d7_1 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d7_1)
    d7_2 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(8, 8), padding='same')(
        Concatenate_waiting[6])
    d7_2 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d7_2)
    d7_3 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(4, 4), padding='same')(
        Concatenate_waiting[6])
    d7_3 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d7_3)
    d7_4 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(2, 2), padding='same')(
        Concatenate_waiting[6])
    d7_4 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d7_4)
    d7_5 = d7_4
    d7_6 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(Concatenate_waiting[6])

    d6 = tf.keras.layers.concatenate([d7_6, e6_6, e5_6, e4_6, e3_6, e2_6, e1_6])
    d6 = tf.keras.layers.Conv2D(filters=base_channel * 7, kernel_size=(3, 3),
                                kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d6)
    d6 = tf.keras.layers.BatchNormalization()(d6)
    d6 = layers.Activation('relu')(d6)
    d6 = tf.keras.layers.Dropout(drop_connect_rate)(d6)
    if Attention:
        d6 = attach_attention_module(d6, Attention)

    d6_5 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(2, 2), padding='same')(d6)
    d6_5 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d6_5)
    d6_4 = d6_5
    d6_3 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(4, 4), padding='same')(d6)
    d6_3 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d6_3)
    d6_2 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(8, 8), padding='same')(d6)
    d6_2 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d6_2)
    d6_1 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(16, 16), padding='same')(d6)
    d6_1 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d6_1)

    d5 = tf.keras.layers.concatenate([d7_5, d6_5, e5_5, e4_5, e3_5, e2_5, e1_5])
    d5 = tf.keras.layers.Conv2D(filters=base_channel * 7, kernel_size=(3, 3),
                                kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d5)
    d5 = tf.keras.layers.BatchNormalization()(d5)
    d5 = layers.Activation('relu')(d5)
    d5 = tf.keras.layers.Dropout(drop_connect_rate)(d5)
    if Attention:
        d5 = attach_attention_module(d5, Attention)

    d5_4 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d5)
    d5_3 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(2, 2), padding='same')(d5)
    d5_3 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d5_3)
    d5_2 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(4, 4), padding='same')(d5)
    d5_2 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d5_2)
    d5_1 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(8, 8), padding='same')(d5)
    d5_1 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d5_1)

    d4 = tf.keras.layers.concatenate([d7_4, d6_4, d5_4, e4_4, e3_4, e2_4, e1_4])
    d4 = tf.keras.layers.Conv2D(filters=base_channel * 7, kernel_size=(3, 3),
                                kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d4)
    d4 = tf.keras.layers.BatchNormalization()(d4)
    d4 = layers.Activation('relu')(d4)
    d4 = tf.keras.layers.Dropout(drop_connect_rate)(d4)
    if Attention:
        d4 = attach_attention_module(d4, Attention)

    d4_3 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(2, 2), padding='same')(d4)
    d4_3 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d4_3)
    d4_2 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(4, 4), padding='same')(d4)
    d4_2 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d4_2)
    d4_1 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(8, 8), padding='same')(d4)
    d4_1 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d4_1)

    d3 = tf.keras.layers.concatenate([d7_3, d6_3, d5_3, d4_3, e3_3, e2_3, e1_3])
    d3 = tf.keras.layers.Conv2D(filters=base_channel * 7, kernel_size=(3, 3),
                                kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d3)
    d3 = tf.keras.layers.BatchNormalization()(d3)
    d3 = layers.Activation('relu')(d3)
    d3 = tf.keras.layers.Dropout(drop_connect_rate)(d3)
    if Attention:
        d3 = attach_attention_module(d3, Attention)

    d3_2 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(2, 2), padding='same')(d3)
    d3_2 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d3_2)
    d3_1 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(4, 4), padding='same')(d3)
    d3_1 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d3_1)

    d2 = tf.keras.layers.concatenate([d7_2, d6_2, d5_2, d4_2, d3_2, e2_2, e1_2])
    d2 = tf.keras.layers.Conv2D(filters=base_channel * 7, kernel_size=(3, 3),
                                kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    d2 = layers.Activation('relu')(d2)
    d2 = tf.keras.layers.Dropout(drop_connect_rate)(d2)
    if Attention:
        d2 = attach_attention_module(d2, Attention)

    d2_1 = tf.keras.layers.Conv2DTranspose(base_channel, (3, 3), strides=(2, 2), padding='same')(d2)
    d2_1 = tf.keras.layers.Conv2D(filters=base_channel, kernel_size=(3, 3),
                                  kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d2_1)
    d1 = tf.keras.layers.concatenate([d7_1, d6_1, d5_1, d4_1, d3_1, d2_1, e1_1])
    d1 = tf.keras.layers.Conv2D(filters=base_channel * 7, kernel_size=(3, 3),
                                kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = layers.Activation('relu')(d1)
    d1 = tf.keras.layers.Dropout(drop_connect_rate)(d1)
    if Attention:
        d1 = attach_attention_module(d1, Attention)


    # # sort layer
    # sort_layer = tf.keras.layers.Dropout(drop_connect_rate)(Concatenate_waiting[6])
    # sort_layer = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(sort_layer)
    # sort_layer = layers.GlobalMaxPooling2D()(sort_layer)
    # sort_layer = layers.Activation('softmax')(sort_layer)
    #
    # sort_layer = tf.expand_dims(tf.expand_dims(sort_layer, axis=1), axis=1)
    #
    # output1 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(d1)
    # output1 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output1)
    # output1 = tf.multiply(output1, sort_layer)
    #
    # output2 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(4, 4), padding='same')(d2)
    # output2 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output2)
    # output2 = tf.multiply(output2, sort_layer)
    #
    # output3 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(8, 8), padding='same')(d3)
    # output3 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output3)
    # output3 = tf.multiply(output3, sort_layer)
    #
    # output4 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(16, 16), padding='same')(d4)
    # output4 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output4)
    # output4 = tf.multiply(output4, sort_layer)
    #
    # output5 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(16, 16), padding='same')(d5)
    # output5 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output5)
    # output5 = tf.multiply(output5, sort_layer)
    #
    # output6 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(32, 32), padding='same')(d6)
    # output6 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output6)
    # output6 = tf.multiply(output6, sort_layer)
    #
    # output7 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(32, 32), padding='same')(Concatenate_waiting[6])
    # output7 = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output7)
    # output7 = tf.multiply(output7, sort_layer)
    #
    # tmp = [output1, output2, output3, output4, output5, output6, output7]
    # outputs = []
    # for item in tmp:
    #     outputs.append(layers.Activation('softmax')(item))
    # model = Model(img_input, outputs, name=model_name)

    output = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(d1)
    output = tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer=CONV_KERNEL_INITIALIZER, padding='same')(output)
    output = layers.Activation('softmax')(output)
    model = Model(img_input, output, name=model_name)

    return model


def efficientnet_b0():
    # https://storage.googleapis.com/keras-applications/efficientnetb0.h5
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.0,
                         dropout_rate=0.2,
                         model_name="efficientnetb0")


def efficientnet_b1():
    # https://storage.googleapis.com/keras-applications/efficientnetb1.h5
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.1,
                         dropout_rate=0.2,
                         model_name="efficientnetb1")


def efficientnet_b2():
    # https://storage.googleapis.com/keras-applications/efficientnetb2.h5
    return efficient_net(width_coefficient=1.1,
                         depth_coefficient=1.2,
                         dropout_rate=0.3,
                         model_name="efficientnetb2")


def efficientnet_b3():
    # https://storage.googleapis.com/keras-applications/efficientnetb3.h5
    return efficient_net(width_coefficient=1.2,
                         depth_coefficient=1.4,
                         dropout_rate=0.3,
                         model_name="efficientnetb3")


def efficientnet_b4():
    # https://storage.googleapis.com/keras-applications/efficientnetb4.h5
    return efficient_net(width_coefficient=1.4,
                         depth_coefficient=1.8,
                         dropout_rate=0.4,
                         model_name="efficientnetb4")


def efficientnet_b5():
    # https://storage.googleapis.com/keras-applications/efficientnetb5.h5
    return efficient_net(width_coefficient=1.6,
                         depth_coefficient=2.2,
                         dropout_rate=0.4,
                         model_name="efficientnetb5")


def efficientnet_b6():
    # https://storage.googleapis.com/keras-applications/efficientnetb6.h5
    return efficient_net(width_coefficient=1.8,
                         depth_coefficient=2.6,
                         dropout_rate=0.5,
                         model_name="efficientnetb6")


def efficientnet_b7():
    # https://storage.googleapis.com/keras-applications/efficientnetb7.h5
    return efficient_net(width_coefficient=2.0,
                         depth_coefficient=3.1,
                         dropout_rate=0.5, )
