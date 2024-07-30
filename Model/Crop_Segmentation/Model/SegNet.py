from tensorflow.keras import layers, Model
import yaml
import tensorflow as tf


with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']
    Attention = yaml_data['Models_Detail']['Unet']['Attention']
    Class_Num = len(yaml_data['Train']['Class_Map'])

# 卷基层初始化方法
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}
IMAGE_ORDERING = "channels_last"

# 全连接层初始化方法
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def vggnet_encoder(input_height=416, input_width=416, pretrained='imagenet'):

    img_input = tf.keras.Input(shape=(input_height, input_width, 3))

    # 416,416,3 -> 208,208,64
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # 208,208,64 -> 128,128,128
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # 104,104,128 -> 52,52,256
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # 52,52,256 -> 26,26,512
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # 26,26,512 -> 13,13,512
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]

# 解码器
def decoder(feature_input, n_classes, n_upSample):
    # feature_input是vggnet第四个卷积块的输出特征矩阵
    # 26,26,512
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(feature_input)
    output = (layers.Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (layers.Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    for _ in range(n_upSample - 2):
        output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
        output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
        output = (layers.Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
        output = (layers.BatchNormalization())(output)

    # 进行一次 UpSampling2D，此时 hw 变为原来的 1/2
    # 208,208,64
    output = (layers.UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (layers.ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (layers.Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(output)
    output = (layers.BatchNormalization())(output)
# 像素级分类层
    # 此时输出为h_input/2,w_input/2,nclasses
    # 208,208,2
    output = layers.Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(output)

    return output

def SegNet(input_shape=(Height, Width, 3),
                  dropout_rate=0.2,
                base_filter = 32,
                  activation="swish",
                  model_name="efficientnet",
                kernel=3,
                pool_size=(2, 2)):
    img_input, features = vggnet_encoder(input_height=input_shape[0], input_width=input_shape[1])
    feature = features[3]  # (26,26,512)
    output = decoder(feature, Class_Num, 4)

    # 将结果进行reshape
    # output = tf.reshape(output, (-1, int(Height / 2) * int(Width / 2), 2))
    output = layers.Softmax()(output)

    model = tf.keras.Model(img_input, output)

    return model
