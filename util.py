import tensorflow as tf
import numpy as np


def connect(input_layer, layers):
    """ Connects the given sequence of layers and returns output layer

    Arguments:
        input_layer (tf.keras.layers.Layer):
            Input layer.
        layers (list of tf.keras.layer.Layer):
            Layers to be connected sequentially.

    Returns:
        output_layer (tf.keras.layer.Layer):
            Output Layer.
    """
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer


def linlogcut(x, a=1e6, b=1e10) -> tf.Tensor:
    """ Function which is linear until a, logarithmic until b and then constant, i.e.
        y = x               x <= a
        y = a + log(x-a+1)  a < x < b
        y = b               b < x
    """
    # Cutoff x after b, this should also cutoff infinities.
    x = tf.where(x < b, x, b * tf.ones(tf.shape(x)))
    # Logarithm after a.
    y = a + tf.where(x < a, x - a, tf.math.log(x - a + 1))
    # Make sure everything is finite.
    y = tf.where(tf.math.is_finite(y), y, b * tf.ones(tf.shape(y)))
    return y


def ensure_shape(x):
    """ Returns input as a 2D array """
    if np.ndim(x) == 2:
        return x                # Return as is.
    if np.ndim(x) == 1:
        return np.array([x])    # Add 1 dimension.
    raise ValueError('Incompatible array with shape: ', np.shape(x))


def distance_matrix_squared(crd1, crd2, dim=2):
    """ Returns the distance matrix or matrices between particles

    Arguments:
        crd1 (np.ndarray): 
            First coordinate set - array or matrix.
        crd2 (np.ndarray): 
            Second coordinate set - array or matrix.
        dim (int):
            Dimension of particle system. 
            If d=2, coordinate vectors are [x1, y1, x2, y2,...]
    """
    crd1 = ensure_shape(crd1)
    crd2 = ensure_shape(crd2)
    n_particles = int(np.shape(crd1)[1]/dim)

    crd1_components = [
        np.tile(np.expand_dims(crd1[:, i::dim], 2), (1, 1, n_particles))
        for i in range(dim)
    ]
    crd2_components = [
        np.tile(np.expand_dims(crd2[:, i::dim], 2), (1, 1, n_particles))
        for i in range(dim)
    ]
    D2_components = [
        (crd1_components[i] - np.transpose(crd2_components[i], axes=(0, 2, 1)))**2
        for i in range(dim)
    ]
    D2 = np.sum(D2_components, axis=0)
    return D2
