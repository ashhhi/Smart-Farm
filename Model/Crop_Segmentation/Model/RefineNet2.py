import tensorflow as tf
import yaml

with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    Attention = yaml_data['Models_Detail']['Unet']['Attention']
    Class_Num = len(yaml_data['Train']['Class_Map'])

def CRPBlock(input_tensor):
    input_tensor = tf.keras.layers.Activation('relu')(input_tensor)
    # Pooling branch 1
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

    # Pooling branch 2
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(pool1)

    # Pooling branch 3
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(pool2)

    # Concatenate pooled features
    concatenated = tf.keras.layers.concatenate([pool1, pool2, pool3], axis=-1)

    # Final feature fusion
    fused = tf.keras.layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same')(concatenated)

    # Residual connection
    output = tf.keras.layers.Add()([input_tensor, fused])

    return output

# 定义基本的卷积块
def Conv2dBlock(inputTensor, base_filters, kernelSize=3, doBatchNorm=True):
    x = tf.keras.layers.Conv2D(filters=base_filters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# 定义RefineNet中的Refine块
def RefineBlock(inputTensor, base_filters, doBatchNorm=True):

    conv1 = Conv2dBlock(inputTensor, base_filters, kernelSize=3, doBatchNorm=doBatchNorm)
    conv2 = Conv2dBlock(conv1, base_filters, kernelSize=3, doBatchNorm=doBatchNorm)
    output = tf.keras.layers.Add()([inputTensor, conv2])
    return output

def RefineNet(input_shape=(Height, Width, 3),
                  dropout_rate=0.2,
                base_filter = 32,
                  activation="swish",
                  model_name="efficientnet",
                doBatchNorm=True):
    inputImage = tf.keras.layers.Input(shape=input_shape)
    # 定义Encoder部分
    c1 = Conv2dBlock(inputImage, base_filter * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((4, 4))(c1)
    p1 = tf.keras.layers.Dropout(dropout_rate)(p1)

    c2 = Conv2dBlock(p1, base_filter * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dropout_rate)(p2)

    c3 = Conv2dBlock(p2, base_filter * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dropout_rate)(p3)

    c4 = Conv2dBlock(p3, base_filter * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropout_rate)(p4)

    # 定义RefineNet部分
    r1 = RefineBlock(p3, base_filter * 4, doBatchNorm=doBatchNorm)
    r2 = RefineBlock(r1, base_filter * 4, doBatchNorm=doBatchNorm)
    r3 = RefineBlock(r2, base_filter * 4, doBatchNorm=doBatchNorm)

    # 定义Decoder部分
    u4 = tf.keras.layers.Conv2DTranspose(base_filter * 2, (3, 3), strides=(2, 2), padding='same')(r3)
    u4 = tf.keras.layers.concatenate([u4, c2])
    u4 = tf.keras.layers.Dropout(dropout_rate)(u4)
    c4 = Conv2dBlock(u4, base_filter * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    u5 = tf.keras.layers.Conv2DTranspose(base_filter * 1, (3, 3), strides=(2, 2), padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, c1])
    u5 = tf.keras.layers.Dropout(dropout_rate)(u5)
    c5 = Conv2dBlock(u5, base_filter * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model

