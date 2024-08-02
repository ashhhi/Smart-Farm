import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def categorical_focal(alpha_=None, gamma=2.0):
    def categorical_focal_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()  # 防止 log(0) 出现
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        # If alpha is a list or a tensor, use it; otherwise, set it as a scalar.
        if alpha_ is None:
            alpha_t = 1.0
        elif isinstance(alpha_, (list, tf.Tensor)):
            alpha = tf.convert_to_tensor(alpha_, dtype=tf.float32)
            alpha_t = tf.reduce_sum(alpha * y_true, axis=-1)
        else:
            alpha_t = alpha_

        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)

        focal_loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)

        mean = tf.reduce_mean(focal_loss)

        tf.summary.scalar('focal loss', mean)

        return mean

    return categorical_focal_fixed

#
# # 生成随机张量
# shape = (2, 224, 320 ,3)  # 张量的形状
# minval = 0  # 随机数的最小值
# maxval = 1  # 随机数的最大值
#
# # 生成第一个张量
# tensor1 = tf.random.uniform(shape, minval=minval, maxval=maxval)
#
# # 生成第二个张量
# tensor2 = tf.random.uniform(shape, minval=minval, maxval=maxval)
#
# # 打印生成的张量
# print("Tensor 1:")
# print(tensor1)
#
# print("\nTensor 2:")
# print(tensor2)
#
# print(categorical_focal(alpha_=[0.25, 0.25, 0.25])(tensor1, tensor2))
