from tensorflow.keras.layers import SeparableConv2D, Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation

# 卷基层初始化方法
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}


def ConvBlock(inputs, filters, kernel_size=3, strides=1, padding='SAME', bn=True, ac=True, activation='relu'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
    if bn:
        x = BatchNormalization()(x)
    if ac:
        x = Activation(activation)(x)
    return x

def SeparableConvBlock(inputs, filters, kernel_size=3, strides=1, padding='SAME', bn=True, ac=True, activation='relu'):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
    if bn:
        x = BatchNormalization()(x)
    if ac:
        x = Activation(activation)(x)
    return x