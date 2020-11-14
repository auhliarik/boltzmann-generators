import tensorflow as tf
import numpy as np
import numbers
import copy


def nonlinear_transform(output_size, n_layers=3, n_hidden=100, activation='relu',
                        init_outputs=None, **args):
    """ Generic dense trainable neural nonlinear transform

    Returns a list of the layers of a dense feedforward network with
    n_layers-1 hidden layers with n_hidden neurons and the specified activation
    functions. The last layer is always linear in order to access the full real
    number range and has output_size output neurons.

    Parameters
    ----------
    output_size : int
        number of output neurons
    n_layers : int
        number of layers, including the linear output layer. n_layers=3 means two
        hidden layers with nonlinear activation and one linear output layer.
    n_hidden : int
        number of neurons in each hidden layer, either a number or an array of
        length n_layers-1 to specify the width of each hidden layer
    activation : str
        nonlinear activation function in hidden layers
    init_outputs : None or float or array
        None means default initialization for the output layer, otherwise
        it is initialized with 0
    **args : kwargs
        Additional keyword arguments passed to the layer

    """
    if isinstance(n_hidden, numbers.Integral):
        n_hidden = n_hidden * np.ones(n_layers-1, dtype=int)
    else:
        n_hidden = np.array(n_hidden)
        if n_hidden.size != n_layers-1:
            raise ValueError('Illegal size of n_hidden. Expecting 1d array with n_layers-1 elements')

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
    def __init__(self, indices, **kwargs):
        """ Returns [:, indices]."""
        self.indices = indices
        super().__init__(**kwargs)

    def call(self, x):
        # Gathering part of an input defined by self.indices
        return tf.gather(x, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.indices.size
