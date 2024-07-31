import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    Attention = yaml_data['Models_Detail']['Unet']['Attention']
    Class_Num = len(yaml_data['Train']['Class_Map'])


def adaptive_max_pool2d(x, output_size):
    input_shape = x.shape
    input_height = input_shape[1]
    input_width = input_shape[2]

    target_height = output_size[0]
    target_width = output_size[1]

    stride_height = input_height // target_height
    stride_width = input_width // target_width
    pooled = tf.keras.layers.MaxPooling2D((stride_height, stride_width))(x)
    return pooled

# defining Conv2d block for our u-net
# this block essentially performs 2 convolution
def ResNetBlock(inputTensor, numFilters, kernelSize=3, residual_path=False):

    # first Conv
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Second Conv
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    residual = inputTensor
    if residual_path:
        residual = Conv2D(filters=numFilters,kernel_size=(1,1),strides=1,padding='same',use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        x = tf.keras.layers.Activation('relu')(x+residual)

    return x

# Now defining Unet
def Unet(input_shape=(Height, Width, 3), dropouts=0.2, numFilters=16):
    img_input = tf.keras.layers.Input(shape=input_shape)
    # defining encoder Path
    e1 = ResNetBlock(img_input, numFilters * 1, kernelSize=3)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(e1)
    p1 = tf.keras.layers.Dropout(dropouts)(p1)
    e1_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                 kernel_initializer='he_normal', padding='same')(e1)
    e1_2 = tf.keras.layers.MaxPooling2D((2, 2))(e1)
    e1_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(e1_2)
    e1_3 = tf.keras.layers.MaxPooling2D((4, 4))(e1)
    e1_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                               kernel_initializer='he_normal', padding='same')(e1_3)
    e1_4 = tf.keras.layers.MaxPooling2D((8, 8))(e1)
    e1_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(e1_4)


    e2 = ResNetBlock(p1, numFilters * 2, kernelSize=3)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(e2)
    p2 = tf.keras.layers.Dropout(dropouts)(p2)
    e2_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                 kernel_initializer='he_normal', padding='same')(e2)
    e2_3 = tf.keras.layers.MaxPooling2D((2, 2))(e2)
    e2_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                 kernel_initializer='he_normal', padding='same')(e2_3)
    e2_4 = tf.keras.layers.MaxPooling2D((4, 4))(e2)
    e2_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(e2_4)

    e3 = ResNetBlock(p2, numFilters * 4, kernelSize=3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(e3)
    p3 = tf.keras.layers.Dropout(dropouts)(p3)
    e3_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                 kernel_initializer='he_normal', padding='same')(e3)
    e3_4 = tf.keras.layers.MaxPooling2D((2, 2))(e3)
    e3_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(e3_4)

    e4 = ResNetBlock(p3, numFilters * 8, kernelSize=3)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(e4)
    p4 = tf.keras.layers.Dropout(dropouts)(p4)
    e4_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                 kernel_initializer='he_normal', padding='same')(e4)

    e5 = ResNetBlock(p4, numFilters * 16, kernelSize=3)

    # defining decoder path
    d5_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(16, 16), padding='same')(e5)
    d5_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d5_1)
    d5_2 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(8, 8), padding='same')(e5)
    d5_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d5_2)
    d5_3 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(4, 4), padding='same')(e5)
    d5_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d5_3)
    d5_4 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(e5)
    d5_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d5_4)

    d4 = tf.keras.layers.concatenate([d5_4, e4_4, e3_4, e2_4, e1_4])
    d4 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d4)
    d4 = tf.keras.layers.BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = tf.keras.layers.Dropout(dropouts)(d4)

    d4_3 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(d4)
    d4_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d4_3)
    d4_2 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(4, 4), padding='same')(d4)
    d4_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d4_2)
    d4_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(8, 8), padding='same')(d4)
    d4_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d4_1)

    d3 = tf.keras.layers.concatenate([d5_3, d4_3, e3_3, e2_3, e1_3])
    d3 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                kernel_initializer='he_normal', padding='same')(d3)
    d3 = tf.keras.layers.BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = tf.keras.layers.Dropout(dropouts)(d3)
    d3_2 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(d3)
    d3_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d3_2)
    d3_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(4, 4), padding='same')(d3)
    d3_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d3_1)

    d2 = tf.keras.layers.concatenate([d5_2, d4_2, d3_2, e2_2, e1_2])
    d2 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                kernel_initializer='he_normal', padding='same')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = tf.keras.layers.Dropout(dropouts)(d2)
    d2_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(d2)
    d2_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                  kernel_initializer='he_normal', padding='same')(d2_1)

    d1 = tf.keras.layers.concatenate([d5_1, d4_1, d3_1, d2_1, e1_1])
    d1 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                kernel_initializer='he_normal', padding='same')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = tf.keras.layers.Dropout(dropouts)(d1)

    # # 深监督
    # # 分类引导模块
    # sort_layer = tf.keras.layers.Dropout(droupouts)(e5)
    # sort_layer = tf.keras.layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(sort_layer)
    # sort_layer = adaptive_max_pool2d(sort_layer, [1, 1])
    # sort_layer = Activation('sigmoid')(sort_layer)
    #
    output1 = tf.keras.layers.Conv2D(Class_Num, (3, 3), kernel_initializer='he_normal', padding='same')(d1)
    # output1 = tf.multiply(output1, sort_layer)
    # output2 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d2)
    # output2 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(output2)
    # output2 = tf.multiply(output2, sort_layer)
    # output3 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d3)
    # output3 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(4, 4), padding='same')(output3)
    # output3 = tf.multiply(output3, sort_layer)
    # output4 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d4)
    # output4 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(8, 8), padding='same')(output4)
    # output4 = tf.multiply(output4, sort_layer)
    # output5 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(e5)
    # output5 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(16, 16), padding='same')(output5)
    # output5 = tf.multiply(output5, sort_layer)


    # tmp = [output2, output3, output4, output5, output1]
    # outputs = []
    # for item in tmp:
    #     outputs.append(Activation('sigmoid')(item))
    # model = tf.keras.Model(inputs=[inputImage], outputs=outputs)
    model = tf.keras.Model(inputs=img_input, outputs=output1)
    return model