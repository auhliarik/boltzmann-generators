import tensorflow as tf
import numpy as np


def connect(input_layer, layers):
    """ Connect the given sequence of layers and returns output layer

    Parameters
    ----------
    input_layer : tf.keras layer
        Input layer
    layers : list of tf.keras layers
        Layers to be connected sequentially

    Returns
    -------
    output_layer : tf.kears layer
        Output Layer
    """
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer


def linlogcut(x, a=100, b=1000):
    """ Function which is linear until a, logarithmic until b and then constant, i.e.
        y = x               x <= a
        y = a + log(x-a+1)  a < x < b
        y = b               b < x
    """
    # Cutoff x after b, this should also cutoff infinities
    x = tf.where(x < b, x, b * tf.ones(tf.shape(x)))
    return x
    # Logarithm after a
    y = a + tf.where(x < a, x - a, tf.math.log(x - a + 1))
    # Make sure everything is finite
    y = tf.where(tf.math.is_finite(y), y, b * tf.ones(tf.shape(y)))
    return y


def ensure_shape(X):
    if np.ndim(X) == 2:
        return X
    if np.ndim(X) == 1:
        return np.array([X])
    raise ValueError('Incompatible array with shape: ', np.shape(X))
