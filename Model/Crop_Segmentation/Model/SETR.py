"""
refer to:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import tensorflow as tf
import yaml
from tensorflow.keras import Model, layers, initializers
import numpy as np

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


with open('config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Image']['Width']
    Height = yaml_data['Image']['Height']
    Attention = yaml_data['Train']['Module']['Attention']

def PatchEmbed(inputs, patch_size=16, embed_dim=768):
    """
        2D Image to Patch Embedding
    """
    B, H, W, C = inputs.shape
    grid_size = (H // patch_size, W // patch_size)
    num_patches = grid_size[0] * grid_size[1]

    x = layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                      strides=patch_size, padding='SAME',
                      kernel_initializer=initializers.LecunNormal(),
                      bias_initializer=initializers.Zeros())(inputs)
    x = tf.reshape(x, [-1, num_patches, embed_dim])
    return x, num_patches


def ConcatClassTokenAddPosEmbed(inputs, embed_dim=768, num_patches=196):
    batch_size = tf.shape(inputs)[0]
    cls_token = tf.Variable(initial_value=tf.zeros([1, 1, embed_dim]),
                            trainable=True,
                            dtype=tf.float32)
    # [1, 1, 768] -> [B, 1, 768]
    cls_token = tf.broadcast_to(cls_token, shape=[batch_size, 1, embed_dim])
    x = tf.concat([cls_token, inputs], axis=1)  # [B, 197, 768]

    pos_embed = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, embed_dim], stddev=0.02),
                            trainable=True,
                            dtype=tf.float32)

    x = tf.add(x, pos_embed)

    return x


def Attention(inputs, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.Zeros()

    _, N, C = inputs.shape

    head_dim = dim // num_heads
    scale = qk_scale or head_dim ** -0.5
    # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
    qkv = layers.Dense(dim * 3, use_bias=qkv_bias,
                       kernel_initializer=k_ini, bias_initializer=b_ini)(inputs)
    # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
    qkv = tf.reshape(qkv, [-1, N, 3, num_heads, C // num_heads])
    # transpose: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
    qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
    # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
    q, k, v = qkv[0], qkv[1], qkv[2]
    # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
    # multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
    attn = tf.matmul(a=q, b=k, transpose_b=True) * scale
    attn = tf.nn.softmax(attn, axis=-1)
    attn = layers.Dropout(attn_drop_ratio)(attn)

    # multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
    x = tf.matmul(attn, v)
    # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
    x = tf.transpose(x, [0, 2, 1, 3])
    # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
    x = tf.reshape(x, [-1, N, C])

    x = layers.Dense(dim,
                     kernel_initializer=k_ini, bias_initializer=b_ini)(x)
    x = layers.Dropout(proj_drop_ratio)(x)

    return x


def MLP(inputs, in_features, mlp_ratio=4.0, drop=0.):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)
    x = layers.Dense(int(in_features * mlp_ratio),
                     kernel_initializer=k_ini, bias_initializer=b_ini)(inputs)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(in_features,
                     kernel_initializer=k_ini, bias_initializer=b_ini)(x)
    x = layers.Dropout(drop)(x)

    return x


def Block(inputs, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
          drop_path_ratio=0.):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = Attention(x, dim, num_heads=num_heads,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
    if drop_path_ratio > 0.:
        x = tf.add(layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1))(x), inputs)
    else:
        x = tf.add(layers.Activation("linear")(x), inputs)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = MLP(x, dim, drop=drop_ratio)
    if drop_path_ratio > 0.:
        x = tf.add(x, layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1))(x))
    else:
        x = tf.add(x, layers.Activation("linear")(x))
    return x


def VisionTransformer(inputs, patch_size=16, embed_dim=768,
                      depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                      drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                      representation_size=None, num_classes=1000, ):
    x, num_patches = PatchEmbed(inputs, patch_size=patch_size, embed_dim=embed_dim)
    x = ConcatClassTokenAddPosEmbed(x, embed_dim=embed_dim, num_patches=num_patches)
    x = layers.Dropout(drop_ratio)(x)
    dpr = np.linspace(0., drop_path_ratio, depth)
    for i in range(depth):
        x = Block(x, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    if representation_size:
        x = layers.Dense(representation_size, activation="tanh")(x)
    else:
        x = layers.Activation("linear")(x)
    x = layers.Dense(num_classes, kernel_initializer=initializers.Zeros())(x)


    return x

def Decoder(input, filters, stride, dropout_rate):
    x = layers.Conv2DTranspose(filters, (3, 3), strides=(stride, stride), padding='same')(input)
    x = layers.Conv2D(filters,
                      kernel_size=3,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def SETR(inputs, patch_size, embed_dim, depth, num_heads,representation_size, num_classes: int = 768, has_logits: bool = True):
    # encoder
    x = VisionTransformer(inputs=inputs,
                          patch_size=patch_size,
                          embed_dim=embed_dim,
                          depth=depth,
                          num_heads=num_heads,
                          representation_size=representation_size if has_logits else None,
                          num_classes=num_classes)

    # remove cls token
    x = x[:, 1:, :]
    x = tf.reshape(x, [-1, inputs.shape[1] // 16, inputs.shape[2] // 16, 3])
    # decoder
    x = Decoder(x, 768, 2, 0.3)
    x = Decoder(x, 768, 2, 0.3)
    x = Decoder(x, 256, 2, 0.3)
    x = Decoder(x, 3, 2, 0.3)

    x = layers.Conv2D(filters=3, kernel_size=1, padding='SAME')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softmax')(x)

    model = Model(inputs, x)

    return model
def vit_base_patch16_224_in21k(input_shape=(Height, Width, 3), num_classes: int = 3, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    img_input = layers.Input(shape=input_shape)
    model = SETR(img_input,patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, num_classes=num_classes, has_logits=has_logits)

    return model


def vit_base_patch32_224_in21k(input_shape=(Height, Width, 3), num_classes: int = 3, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    img_input = layers.Input(shape=input_shape)
    model = SETR(img_input, patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768,
                 num_classes=num_classes, has_logits=has_logits)

    return model


def vit_large_patch16_224_in21k(input_shape=(Height, Width, 3), num_classes: int = 3, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """

    img_input = layers.Input(shape=input_shape)
    model = SETR(img_input, patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=1024,
                 num_classes=num_classes, has_logits=has_logits)

    return model


def vit_large_patch32_224_in21k(input_shape=(Height, Width, 3), num_classes: int = 3, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    img_input = layers.Input(shape=input_shape)
    model = SETR(img_input, patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024,
                 num_classes=num_classes, has_logits=has_logits)
    return model


def vit_huge_patch14_224_in21k(input_shape=(Height, Width, 3), num_classes: int = 3, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    img_input = layers.Input(shape=input_shape)
    model = SETR(img_input, patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280,
                 num_classes=num_classes, has_logits=has_logits)
    return model
