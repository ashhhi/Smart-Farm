import numpy as np
import tensorflow as tf

# Binary Classification
def iou_metric(y_true, y_pred):
    # 将预测结果转换为二值化（0 或 1）
    y_pred = tf.round(y_pred)
    # print(y_pred)
    # print(y_true)

    # 计算交集
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))

    # 计算并集
    union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection

    # 计算 IoU
    iou = intersection / union
    return iou

# Multipul Classification
def mIoU(y_true, y_pred):
    true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
    pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
    cm = tf.math.confusion_matrix(true, pred, dtype=tf.float32)
    diag_item = tf.linalg.diag_part(cm)
    # return IoU_per
    mIoU = tf.reduce_mean(diag_item / (tf.reduce_sum(cm, 0) + tf.reduce_sum(cm, 1) - tf.linalg.diag_part(cm)))

    # tf.summary.scalar('mean IoU', mIoU)


    return mIoU

#
# gt = tf.constant(
# [[
#     [[0,1,0],[0,1,0],[0,1,0]],
#     [[0,0,1],[0,0,1],[1,0,0]],
#     [[1,0,0],[1,0,0],[0,1,0]]
# ]], dtype=tf.float32
# )
#
# pred = tf.constant(
# [[
#     [[0.3,0.5,0.2],[0.1,0.3,0.6],[0.2,0.5,0.3]],
#     [[0.8,0.1,0.1],[0.3,0.4,0.3],[0.2,0.3,0.5]],
#     [[0.9,0.05,0.05],[0.4,0.4,0.2],[0.1,0.2,0.7]]
# ]]
# )
#
# # print(tf.argmax(pred, -1))
# # print(tf.argmax(gt, -1))
# #
# print(mIoU(gt, pred))
