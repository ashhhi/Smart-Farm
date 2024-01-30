import tensorflow as tf

def mean_relative_error(y_true, y_pred):
    relative_error = tf.abs((y_true - y_pred) / y_true)
    mean_relative_error = tf.reduce_mean(relative_error)
    return mean_relative_error

