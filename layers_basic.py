import tensorflow as tf
import numpy as np
import numbers
import copy


def nonlinear_transform(output_size, n_layers=3, n_hidden=100, activation='relu',
                        init_outputs=None, **args):
    """ Returns layers for a generic dense trainable neural nonlinear transform

    Arguments:
        output_size (int):
            Number of output neurons.
        n_layers (int):
            Number of hidden layers of the dense NN.
        n_hidden (int or list):
            Number of neurons (units) in each hidden layer, either a number
            or an array of length n_layers to specify the width of individual hidden layers.
        activation (str):
            Nonlinear activation function used in the hidden layers.
        init_outputs (None or float or array):
            None means default initialization for the output layer, otherwise
            it is initialized with vector of given constants.
        args (kwargs):
            Additional keyword arguments passed to the layer.

    Returns:
        list[tf.keras.layers.Dense]:
            List of the layers for a dense feedforward neural network that consists of
            - n_layers hidden layers with n_hidden neurons using the specified
              activation function and
            - the last layer which is always linear (without the act. func.)
              in order to access the full range of the real numbers
              and has output_size output neurons.
    """
    if isinstance(n_hidden, numbers.Integral):
        n_hidden = n_hidden * np.ones(n_layers, dtype=int)
    else:
        n_hidden = np.array(n_hidden)
        if n_hidden.size != n_layers:
            raise ValueError("Illegal size of n_hidden. Expecting 1D array with n_layers elements")

    neural_network_layers = [tf.keras.layers.Dense(nh, activation=activation, **args) for nh in n_hidden]

    if init_outputs is None:
        final_layer = tf.keras.layers.Dense(output_size, activation='linear', **args)
    else:
        argscopy = copy.deepcopy(args)
        argscopy['kernel_initializer'] = tf.keras.initializers.Zeros()
        argscopy['bias_initializer'] = tf.keras.initializers.Constant(init_outputs)
        final_layer = tf.keras.layers.Dense(output_size, activation='linear', **argscopy)
    neural_network_layers += [final_layer]

    return neural_network_layers


class IndexLayer(tf.keras.layers.Layer):
    """ Returns[:, indices] """
    def __init__(self, indices, **kwargs):
        self.indices = indices
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        # Gathering part of an input defined by self.indices
        return tf.gather(x, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.indices.size
