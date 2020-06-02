import tensorflow as tf
import numpy as np

from layers_basic import IndexLayer
from util import connect

def split_merge_indices(ndim, nchannels=2, channels=None):
    """
    Parameters:
    -----------
    ndim : int
        Dimension of vector x/z, i.e. vector x/z has shape (ndim,).
    nchannels : int
        Number of channels into which vector should be split.
        Does not have to be specified if channels are given.
    channels : array
        Array with number of channel for each coordinate.

    Returns:
    --------
    channels : array
        See 'channels' in parameters section.
    indices_split : array
        Array containg 'nchannels' arrays - one for each channel with indices
        of coordinates going into this channel.
    indices_merge : array
        Array containing indices of coordinates, as they appear in an array that
        was created by concatenating coordinate arrays, that have been split
        into channels using 'indices_split'.
    """
    if channels is None:
        channels = np.tile(np.arange(nchannels), int(ndim/nchannels)+1)[:ndim]
    else:
        channels = np.array(channels)
        nchannels = np.max(channels) + 1
    indices_split = []
    idx = np.arange(ndim)
    for c in range(nchannels):
        isplit = np.where(channels == c)[0]
        indices_split.append(isplit)
    indices_merge = np.concatenate(indices_split).argsort()
    return channels, indices_split, indices_merge

class SplitChannels(object):
    def __init__(self, ndim, nchannels=2, channels=None):
        """ Splits channels forward and merges them backward. """
        ch, s, m = split_merge_indices(ndim, nchannels=nchannels, channels=channels)
        self.channels = ch
        self.indices_split = s
        self.indices_merge = m

    @classmethod
    def from_dict(cls, D):
        channels = D['channels']
        dim = channels.size
        nchannels = channels.max() + 1
        return cls(dim, nchannels=nchannels, channels=channels)

    def to_dict(self):
        D = {}
        D['channels'] = self.channels
        return D

    def connect_xz(self, x):
        # split x into different coordinate channels
        self.output_z = [IndexLayer(isplit)(x) for isplit in self.indices_split]
        return self.output_z

    def connect_zx(self, z):
        # first concatenate
        x_scrambled = tf.keras.layers.Concatenate()(z)
        # unscramble x
        self.output_x = IndexLayer(self.indices_merge)(x_scrambled)
        return self.output_x

class MergeChannels(SplitChannels):
    """ Merges channels forward and splits them backward. """
    def connect_xz(self, x):
        # first concatenate
        z_scrambled = tf.keras.layers.Concatenate()(x)
        # unscramble x
        self.output_z = IndexLayer(self.indices_merge)(z_scrambled) # , name='output_z'
        return self.output_z

    def connect_zx(self, z):
        # split X into different coordinate channels
        self.output_x = [IndexLayer(isplit)(z) for isplit in self.indices_split]
        return self.output_x

class CompositeLayer(object):
    def __init__(self, transforms):
        """ Composite layer consisting of multiple keras layers with shared parameters """
        self.transforms = transforms

    # Note: util.deserialize_layers and util.serialize_layers must be added to
    # my code in order to make these methods work.

    # @classmethod
    # def from_dict(cls, d):
    #     from deep_boltzmann.networks.util import deserialize_layers
    #     transforms = deserialize_layers(d['transforms'])
    #     return cls(transforms)
    #
    # def to_dict(self):
    #     from deep_boltzmann.networks.util import serialize_layers
    #     D = {}
    #     D['transforms'] = serialize_layers(self.transforms)
    #     return D

class RealNVP(CompositeLayer):
    def __init__(self, transforms):
        """ Two sequential Real NVP transformations and their inverse transformations.

        Parameters
        ----------
        transforms : list
            List [S1, T1, S2, T2] with keras layers for scaling and translation transforms

        """
        super().__init__(transforms)
        self.S1 = transforms[0]
        self.T1 = transforms[1]
        self.S2 = transforms[2]
        self.T2 = transforms[3]

    def connect_xz(self, x):
        def lambda_exp(x):
            return tf.exp(x)
        def lambda_sum(x):
            return tf.reduce_sum(x[0], axis=1) + tf.reduce_sum(x[1], axis=1)

        x1 = x[0]
        x2 = x[1]
        self.input_x1 = x1
        self.input_x2 = x2

        y1 = x1
        self.Sxy_layer = connect(x1, self.S1)
        self.Txy_layer = connect(x1, self.T1)
        prodx = tf.keras.layers.Multiply()([x2, tf.keras.layers.Lambda(lambda_exp)(self.Sxy_layer)])
        y2 = tf.keras.layers.Add()([prodx, self.Txy_layer])

        self.output_z2 = y2
        self.Syz_layer = connect(y2, self.S2)
        self.Tyz_layer = connect(y2, self.T2)
        prody = tf.keras.layers.Multiply()([y1, tf.keras.layers.Lambda(lambda_exp)(self.Syz_layer)])
        self.output_z1 = tf.keras.layers.Add()([prody, self.Tyz_layer])

        # log det(dz/dx)
        self.log_det_xz = tf.keras.layers.Lambda(lambda_sum)([self.Sxy_layer, self.Syz_layer])

        return [self.output_z1, self.output_z2] + x[2:]  # append other layers if there are any

    def connect_zx(self, z):
        def lambda_negexp(x):
            return tf.exp(-x)
        def lambda_negsum(x):
            return tf.reduce_sum(-x[0], axis=1) + tf.reduce_sum(-x[1], axis=1)

        z1 = z[0]
        z2 = z[1]
        self.input_z1 = z1
        self.input_z2 = z2

        y2 = z2
        self.Szy_layer = connect(z2, self.S2)
        self.Tzy_layer = connect(z2, self.T2)
        z1_m_Tz2 = tf.keras.layers.Subtract()([z1, self.Tzy_layer])
        y1 = tf.keras.layers.Multiply()([z1_m_Tz2, tf.keras.layers.Lambda(lambda_negexp)(self.Szy_layer)])

        self.output_x1 = y1
        self.Syx_layer = connect(y1, self.S1)
        self.Tyx_layer = connect(y1, self.T1)
        y2_m_Ty1 = tf.keras.layers.Subtract()([y2, self.Tyx_layer])
        self.output_x2 = tf.keras.layers.Multiply()([y2_m_Ty1, tf.keras.layers.Lambda(lambda_negexp)(self.Syx_layer)])

        # log det(dx/dz)
        # TODO: check Jacobian
        self.log_det_zx = tf.keras.layers.Lambda(lambda_negsum)([self.Szy_layer, self.Syx_layer])

        return [self.output_x1, self.output_x2] + z[2:]  # append other layers if there are any

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Shape is (batchsize,) """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Shape is (batchsize,) """
        return self.log_det_zx
