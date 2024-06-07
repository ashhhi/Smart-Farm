import itertools
import numpy as np
import tensorflow as tf


def generate_anchors(image_shape,
                     pyramid_levels=[3, 4, 5, 6, 7],
                     anchor_scale=4.,
                     scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
                     ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]):
    """
    生成用于目标检测的anchor boxes

    Args:
        image_shape (tuple): 输入图像的形状 (height, width)
        pyramid_levels (list): 特征金字塔的层级
        anchor_scale (float): anchor box的缩放因子
        scales (np.ndarray): anchor box的尺度
        ratios (list): anchor box的宽高比

    Returns:
        anchor_boxes (tf.Tensor): shape为 (num_anchors, 4) 的anchor box坐标张量
    """
    strides = [2 ** x for x in pyramid_levels]

    boxes_all = []
    for stride in strides:
        if image_shape[1] % stride != 0:
            raise ValueError('input size must be divided by the stride.')

        base_anchor_size = anchor_scale * stride

        # 遍历不同的尺度和宽高比
        for scale, ratio in itertools.product(scales, ratios):
            anchor_size_x = base_anchor_size * ratio[0]
            anchor_size_y = base_anchor_size * ratio[1]

            x = tf.range(stride / 2, image_shape[1], stride, dtype=tf.float32)
            y = tf.range(stride / 2, image_shape[0], stride, dtype=tf.float32)

            xv, yv = tf.meshgrid(x, y)
            xv = tf.reshape(xv,[-1])
            yv = tf.reshape(yv,[-1])

            # 计算anchor box的坐标
            x1 = xv - anchor_size_x / 2
            y1 = yv - anchor_size_y / 2
            x2 = xv + anchor_size_x / 2
            y2 = yv + anchor_size_y / 2

            boxes = tf.stack([y1, x1, y2, x2], axis=-1)
            boxes_all.append(boxes)

    anchor_boxes = tf.concat(boxes_all, axis=0)
    anchor_boxes = tf.expand_dims(anchor_boxes, axis=0)

    return anchor_boxes