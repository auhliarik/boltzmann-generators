import tensorflow as tf
import numpy as np

from layers_basic import IndexLayer
from util import connect


def split_merge_indices(n_dim, n_channels=2, channels=None):
    """ Set indices for channels in Real NVP layers
    Arguments:
        n_dim (int):
            Dimension of vector x/z, i.e. vector x/z has shape (n_dim,).
        n_channels (int):
            Number of channels into which vector should be split.
            Does not have to be specified if channels are given.
        channels (np.ndarray):
            Array with number of channel for each coordinate.

    Returns:
        channels (array):
            See 'channels' in parameters section.
        indices_split (array):
            Array containing 'n_channels' arrays - one for each channel with indices
            of coordinates going into this channel.
        indices_merge (array):
            Array containing indices of coordinates, as they appear in an array that
            was created by concatenating coordinate arrays, that have been split
            into channels using 'indices_split'.
    """
    if channels is None:
        channels = np.tile(np.arange(n_channels), int(n_dim/n_channels)+1)[:n_dim]
    else:
        channels = np.array(channels)
        n_channels = np.max(channels) + 1

    indices_split = []
    for c in range(n_channels):
        i_split = np.where(channels == c)[0]
        indices_split.append(i_split)
    indices_merge = np.concatenate(indices_split).argsort()
    return channels, indices_split, indices_merge


class SplitChannels:
    def __init__(self, n_dim, n_channels=2, channels=None, name_of_merge=None):
        """ Splits channels forward and merges them backward. """
        ch, s, m = split_merge_indices(n_dim, n_channels=n_channels, channels=channels)
        self.channels = ch
        self.indices_split = s
        self.indices_merge = m
        self.name_of_merge = name_of_merge

        self.output_z = None
        self.output_x = None

    def connect_xz(self, x):
        # split x into different coordinate channels
        self.output_z = [IndexLayer(i_split)(x) for i_split in self.indices_split]
        return self.output_z

    def connect_zx(self, z):
        # first concatenate
        x_scrambled = tf.keras.layers.Concatenate()(z)
        # unscramble x
        self.output_x = IndexLayer(self.indices_merge, name=self.name_of_merge)(x_scrambled)
        return self.output_x


class MergeChannels(SplitChannels):
    """ Merges channels forward and splits them backward. """
    def connect_xz(self, x):
        # first concatenate
        z_scrambled = tf.keras.layers.Concatenate()(x)
        # unscramble x
        self.output_z = IndexLayer(self.indices_merge, name=self.name_of_merge)(z_scrambled)
        return self.output_z

    def connect_zx(self, z):
        # split X into different coordinate channels
        self.output_x = [IndexLayer(i_split)(z) for i_split in self.indices_split]
        return self.output_x


class CompositeLayer:
    def __init__(self, transforms):
        """ Composite layer consisting of multiple keras layers with shared parameters """
        self.transforms = transforms


class RealNVP(CompositeLayer):
    """ Two sequential Real NVP transformations and their inverse transformations. """
    def __init__(self, transforms):
        """
        Arguments:
            transforms (list):
                List [S1, T1, S2, T2] with keras layers for scaling and translation transforms.
        """
        super().__init__(transforms)
        self.S1 = transforms[0]
        self.T1 = transforms[1]
        self.S2 = transforms[2]
        self.T2 = transforms[3]

        # Other attributes to be filled later
        # Inputs and outputs for x -> z
        self.input_x1 = None
        self.input_x2 = None
        self.output_z1 = None
        self.output_z2 = None
        self.log_det_xz = None

        # Inputs and outputs for z -> x
        self.input_z1 = None
        self.input_z2 = None
        self.output_x1 = None
        self.output_x2 = None
        self.log_det_zx = None

        self.Sxy_layer = None
        self.Txy_layer = None
        self.Syz_layer = None
        self.Tyz_layer = None

        self.Szy_layer = None
        self.Tzy_layer = None
        self.Syx_layer = None
        self.Tyx_layer = None

    def connect_xz(self, x):
        def lambda_exp(x_in):
            return tf.exp(x_in)

        def lambda_sum(x_in):
            return tf.reduce_sum(x_in[0], axis=1) + tf.reduce_sum(x_in[1], axis=1)

        x1 = x[0]
        x2 = x[1]
        self.input_x1 = x1
        self.input_x2 = x2

        y1 = x1
        self.Sxy_layer = connect(x1, self.S1)
        self.Txy_layer = connect(x1, self.T1)
        product_x = tf.keras.layers.Multiply()(
            [x2, tf.keras.layers.Lambda(lambda_exp)(self.Sxy_layer)]
        )
        y2 = tf.keras.layers.Add()([product_x, self.Txy_layer])

        self.output_z2 = y2
        self.Syz_layer = connect(y2, self.S2)
        self.Tyz_layer = connect(y2, self.T2)
        product_y = tf.keras.layers.Multiply()(
            [y1, tf.keras.layers.Lambda(lambda_exp)(self.Syz_layer)]
        )
        self.output_z1 = tf.keras.layers.Add()([product_y, self.Tyz_layer])

        # log det(dz/dx)
        self.log_det_xz = tf.keras.layers.Lambda(lambda_sum)([self.Sxy_layer, self.Syz_layer])

        return [self.output_z1, self.output_z2] + x[2:]  # append other layers if there are any

    def connect_zx(self, z):
        def lambda_negexp(x_in):
            return tf.exp(-x_in)

        def lambda_negsum(x_in):
            return tf.reduce_sum(-x_in[0], axis=1) + tf.reduce_sum(-x_in[1], axis=1)

        z1 = z[0]
        z2 = z[1]
        self.input_z1 = z1
        self.input_z2 = z2

        y2 = z2
        self.Szy_layer = connect(z2, self.S2)
        self.Tzy_layer = connect(z2, self.T2)
        z1_minus_Tz2 = tf.keras.layers.Subtract()([z1, self.Tzy_layer])
        y1 = tf.keras.layers.Multiply()(
            [z1_minus_Tz2, tf.keras.layers.Lambda(lambda_negexp)(self.Szy_layer)]
        )

        self.output_x1 = y1
        self.Syx_layer = connect(y1, self.S1)
        self.Tyx_layer = connect(y1, self.T1)
        y2_minus_Ty1 = tf.keras.layers.Subtract()([y2, self.Tyx_layer])
        self.output_x2 = tf.keras.layers.Multiply()(
            [y2_minus_Ty1, tf.keras.layers.Lambda(lambda_negexp)(self.Syx_layer)]
        )

        # log det(dx/dz)
        self.log_det_zx = tf.keras.layers.Lambda(lambda_negsum)([self.Szy_layer, self.Syx_layer])

        return [self.output_x1, self.output_x2] + z[2:]  # append other layers if there are any

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Shape is (batch_size,) """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Shape is (batch_size,) """
        return self.log_det_zx
