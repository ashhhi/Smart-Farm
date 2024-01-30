import pdb
import numpy as np
import tensorflow as tf

# have issues
def loss1(y_true, y_pred, alpha=1, lamda=1, mju=1):
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    e = abs(y_pred - y_true)
    loss_depth = tf.reduce_mean(tf.math.log(e + alpha))

    e_grad_dx, e_grad_dy = tf.image.image_gradients(e)
    loss_grad_dx = tf.reduce_mean(tf.math.log(abs(e_grad_dx) + alpha))
    loss_grad_dy = tf.reduce_mean(tf.math.log(abs(e_grad_dy) + alpha))
    loss_grad = loss_grad_dx + loss_grad_dy

    gt_grad_dx, gt_grad_dy = tf.image.image_gradients(y_true)
    pred_grad_dx, pred_grad_dy = tf.image.image_gradients(y_pred)
    ones = tf.ones([tf.shape(pred_grad_dx)[0], 1, tf.shape(pred_grad_dx)[2], tf.shape(pred_grad_dx)[3]])
    gt_normal = tf.concat([-gt_grad_dx, -gt_grad_dy, ones], axis=1)
    pred_normal = tf.concat([-pred_grad_dx, -pred_grad_dy, ones], axis=1)
    loss_normal = abs(tf.reduce_mean(1 - tf.keras.losses.CosineSimilarity(axis=1)(gt_normal, pred_normal)))

    loss = loss_depth + lamda * loss_grad + mju * loss_normal

    return loss

def loss2(y_true, y_pred, ssim_loss_weight = 0.85, l1_loss_weight = 0.1, edge_loss_weight = 0.9):
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )

    # Structural similarity (SSIM) index
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    loss = (
            (ssim_loss_weight * ssim_loss)
            + (l1_loss_weight * l1_loss)
            + (edge_loss_weight * depth_smoothness_loss)
    )

    return loss

