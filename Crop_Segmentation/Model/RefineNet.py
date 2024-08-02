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

    # # Concatenate pooled features
    # concatenated = tf.keras.layers.concatenate([pool1, pool2, pool3], axis=-1)
    #
    # # Final feature fusion
    # fused = tf.keras.layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same')(concatenated)

    # Residual connection
    output = tf.keras.layers.Add()([input_tensor, pool1, pool2, pool3])

    return output

# 定义基本的卷积块
def ResConv2dBlock(inputTensor, base_filters, kernelSize=3):
    x = tf.keras.layers.Activation('relu')(inputTensor)
    x = tf.keras.layers.Conv2D(filters=base_filters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=base_filters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    residual = tf.keras.layers.Conv2D(filters=base_filters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)
    x = tf.keras.layers.Add()([residual, x])
    return x

# 定义RefineNet中的Refine块
def RefineNet(input_shape=(Height, Width, 3),
                  dropout_rate=0.2,
                base_filter = 256,
                  activation="swish",
                  model_name="efficientnet",
                doBatchNorm=True):

    img_input = tf.keras.layers.Input(shape=input_shape)
    resized_input_4 = tf.image.resize(img_input, (Height // 4, Width // 4), method=tf.image.ResizeMethod.BILINEAR)
    resized_input_8 = tf.image.resize(img_input, (Height // 8, Width // 8), method=tf.image.ResizeMethod.BILINEAR)
    resized_input_16 = tf.image.resize(img_input, (Height // 16, Width // 16), method=tf.image.ResizeMethod.BILINEAR)
    resized_input_32 = tf.image.resize(img_input, (Height // 32, Width // 32), method=tf.image.ResizeMethod.BILINEAR)

    # RCU
    multi = []
    for layer in [resized_input_4, resized_input_8, resized_input_16, resized_input_32]:
        tmp = ResConv2dBlock(layer, base_filter, kernelSize=3)
        tmp = ResConv2dBlock(tmp, base_filter, kernelSize=3)
        multi.append(tmp)

    # Multi-Resolution Fusion
    fuse8 = tf.keras.layers.Conv2DTranspose(base_filter, (3, 3), strides=(2, 2), padding='same')(multi[1])
    fuse16 = tf.keras.layers.Conv2DTranspose(base_filter, (3, 3), strides=(4, 4), padding='same')(multi[2])
    fuse32 = tf.keras.layers.Conv2DTranspose(base_filter, (3, 3), strides=(8, 8), padding='same')(multi[3])
    fuse = tf.keras.layers.Add()([multi[0], fuse8, fuse16, fuse32])

    # CRP
    crp = CRPBlock(fuse)

    # upsample
    res = tf.keras.layers.Conv2DTranspose(base_filter, (3, 3), strides=(4, 4), padding='same')(crp)
    res = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3),
                               kernel_initializer='he_normal', padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation('softmax')(res)

    model = tf.keras.Model(inputs=[img_input], outputs=[res])
    return model

