import tensorflow as tf
import numpy as np

import losses
from invertible_layers import *
from layers_basic import *

class InvNet(object):

    def __init__(self, dim, layers, prior='normal'):
        """ Stack of invertible layers
        Parameters
        ----------
        dim : int
            Dimension of the (physical) system
        layers : list
            list of invertible layers
        prior : str
            Type of prior - only 'normal' implemented so far

        """

        self.dim = dim
        self.layers = layers
        self.prior = prior
        self.connect_layers()

        # compute total Jacobian for x->z transformation
        log_det_xzs = []
        for l in layers:
            if hasattr(l, 'log_det_xz'):
                log_det_xzs.append(l.log_det_xz)
        if len(log_det_xzs) == 0:
            self.TxzJ = None
        else:
            if len(log_det_xzs) == 1:
                self.log_det_xz = log_det_xzs[0]
            else:
                self.log_det_xz = tf.keras.layers.Add()(log_det_xzs)
            self.TxzJ = tf.keras.Model(inputs=self.input_x, outputs=[self.output_z, self.log_det_xz])

        # compute total Jacobian for z->x transformation
        log_det_zxs = []
        for l in layers:
            if hasattr(l, 'log_det_zx'):
                log_det_zxs.append(l.log_det_zx)
        if len(log_det_zxs) == 0:
            self.TzxJ = None
        else:
            if len(log_det_zxs) == 1:
                self.log_det_zx = log_det_zxs[0]
            else:
                self.log_det_zx = tf.keras.layers.Add()(log_det_zxs)
            self.TzxJ = tf.keras.Model(inputs=self.input_z, outputs=[self.output_x, self.log_det_zx])

    def connect_xz(self, x):
        z = None
        for i in range(len(self.layers)):
            z = self.layers[i].connect_xz(x)  # connect
            #print(self.layers[i])
            #print('Inputs\n', x)
            #print()
            #print('Outputs\n', z)
            #print('------------')
            #print()
            x = z  # rename output
        return z

    def connect_zx(self, z):
        x = None
        for i in range(len(self.layers)-1, -1, -1):
            x = self.layers[i].connect_zx(z)  # connect
            #print(self.layers[i])
            #print('Inputs\n', z)
            #print()
            #print('Outputs\n', x)
            #print('------------')
            #print()
            z = x  # rename output to next input
        return x

    def connect_layers(self):
        # X -> Z
        self.input_x = tf.keras.layers.Input(shape=(self.dim,))
        self.output_z = self.connect_xz(self.input_x)

        # Z -> X
        self.input_z = tf.keras.layers.Input(shape=(self.dim,))
        self.output_x = self.connect_zx(self.input_z)

        # build networks
        self.Txz = tf.keras.Model(inputs=self.input_x, outputs=self.output_z)
        self.Tzx = tf.keras.Model(inputs=self.input_z, outputs=self.output_x)

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        #return self.log_det_xz.output
        log_det_Jxzs = []
        for l in self.layers:
            if hasattr(l, 'log_det_Jxz'):
                log_det_Jxzs.append(l.log_det_Jxz)
        if len(log_det_Jxzs) == 0:
            return tf.ones((self.output_z.shape[0],))
        if len(log_det_Jxzs) == 1:
            return log_det_Jxzs[0]
        return tf.reduce_sum(log_det_Jxzs, axis=0, keepdims=False)

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        #return self.log_det_zx.output
        log_det_Jzxs = []
        for l in self.layers:
            if hasattr(l, 'log_det_Jzx'):
                log_det_Jzxs.append(l.log_det_Jzx)
        if len(log_det_Jzxs) == 0:
            return tf.ones((self.output_x.shape[0],))
        if len(log_det_Jzxs) == 1:
            return log_det_Jzxs[0]
        return tf.reduce_sum(log_det_Jzxs, axis=0, keepdims=False)

    def log_likelihood_z_normal(self, std=1.0):
        """ Returns the log likelihood (except for a constant) of z|x
        assuming a Normal distribution in z
        """
        #return self.log_det_Jxz - self.dim * tf.log(std) - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)
        return self.log_det_Jxz - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)

    def train_ML(self, x, w=1, xval=None, optimizer=None, lr=0.001, clipnorm=None, epochs=2000, batch_size=1024,
                 std=1.0, reg_Jxz=0.0, verbose=1, return_test_energies=False):

        if optimizer is None:
            if clipnorm is None:
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                                     clipnorm=clipnorm)

        if self.prior == 'normal':
            if reg_Jxz == 0:
                self.Txz.compile(optimizer, loss=losses.Loss_ML_normal(self, w)) #loss_weights=[w])
        else:
            raise NotImplementedError('ML for prior ' + self.prior + ' is not implemented.')

        if xval is not None:
            validation_data = (xval, np.zeros_like(xval))
        else:
            validation_data = None

        #hist = self.Txz.fit(x=x, y=np.zeros_like(x), validation_data=validation_data,
        #                    batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True)
        # data preprocessing


        N = x.shape[0]
        I = np.arange(N)
        loss_train = []
        energies_x_val = []
        energies_z_val = []
        loss_val = []
        y = np.zeros((batch_size, self.dim))
        for e in range(epochs):
            # sample batch
            x_batch = tf.gather(x, np.random.choice(I, size=batch_size, replace=True))
            # x_batch = np.random.choice(x, size=batch_size, replace=True)
            l = self.Txz.train_on_batch(x=x_batch, y=y)
            loss_train.append(l)

            # validate
            if xval is not None:
                # xval_batch = xval[np.random.choice(I, size=batch_size, replace=True)]
                xval_batch = tf.gather(xval, np.random.choice(I, size=batch_size, replace=True))
                l = self.Txz.test_on_batch(x=xval_batch, y=y)
                loss_val.append(l)
                if return_test_energies:
                    z = self.sample_z(nsample=batch_size)
                    xout = self.transform_zx(z)
                    energies_x_val.append(self.energy_model.energy(xout))
                    zout = self.transform_xz(xval_batch)
                    energies_z_val.append(self.energy_z(zout))

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                str_ += self.Txz.metrics_names[0] + ' '
                str_ += '{:.4f}'.format(loss_train[-1]) + ' '
                if xval is not None:
                    str_ += '{:.4f}'.format(loss_val[-1]) + ' '
#                for i in range(len(self.Txz.metrics_names)):

                    #str_ += self.Txz.metrics_names[i] + ' '
                    #str_ += '{:.4f}'.format(loss_train[-1][i]) + ' '
                    #if xval is not None:
                    #    str_ += '{:.4f}'.format(loss_val[-1][i]) + ' '
                print(str_)
                sys.stdout.flush()

        if return_test_energies:
            return loss_train, loss_val, energies_x_val, energies_z_val
        else:
            return loss_train #, loss_val

def invnet(dim, layer_types, energy_model=None, channels=None,
           nl_layers=2, nl_hidden=100, nl_layers_scale=None, nl_hidden_scale=None,
           nl_activation='relu', nl_activation_scale='tanh', prior='normal',
           whiten=None, whiten_keepdims=None,
           **layer_args):
    """
    Returns an instance of InvNet or EnergyInvNet class

    Parameters:
    -----------
    dim : int
        Dimension of the (physical) system
    layer_types : str
        String describing the sequence of layers. Usage:
            R RealNVP layer
            r RealNVP layer, share parameters with last layer
        Splitting and merging layers will be added automatically
    energy_model : Energy model class
        Class with energy() and dim
    channels : array or None
        Assignment of dimensions to channels (0/1 array of length ndim)
    nl_layers : int
        Number of hidden layers in the nonlinear transformations
    nl_hidden : int
        Number of hidden units in each nonlinear layer
    nl_activation : str
        Hidden-neuron activation functions used in the nonlinear layers
    nl_activation_scale : str
        Hidden-neuron activation functions used in scaling networks.
        If None, nl_activation will be used.
    prior : str
        Form of the prior distribution
    whiten : None or array
        If not None, compute a whitening transformation with respect to given coordinates
    whiten_keepdims : None or int
        Number of largest-variance dimensions to keep after whitening.
    """
    # fix channels
    channels, indices_split, indices_merge = split_merge_indices(dim,
                                                                 nchannels=2,
                                                                 channels=channels)

    # augment layer types with split and merge layers
    split = False
    tmp = ''
    if whiten is not None:
        tmp += 'W'
    for ltype in layer_types:
        if ltype == 'R' and not split:
            tmp += '<'
            split = True
        tmp += ltype
    if split:
        tmp += '>'
    layer_types = tmp
    print("Layers of invertible NN:", layer_types)

    # prepare layers
    layers = []

    # Properties of scale tranforms are the same as proparties aof translational
    # transforms unless they are specified
    if nl_activation_scale is None:
        nl_activation_scale = nl_activation
    if nl_layers_scale is None:
        nl_layers_scale = nl_layers
    if nl_hidden_scale is None:
        nl_hidden_scale= nl_hidden

    # Number of dimensions left in the signal.
    # The remaining dimensions are going to latent space directly
    dim_L = dim     # number of dimensions in left channel (the default one)
    dim_R = 0       # number of dimensions in right channel
    dim_Z = 0

    # translate and scale layers
    T1 = None
    T2 = None
    S1 = None
    S2 = None

    for ltype in layer_types:
        print(ltype, dim_L, dim_R, dim_Z)
        if ltype == '<':
            if dim_R > 0:
                raise RuntimeError('Already split. Cannot invoke split layer.')
            channels_cur = np.concatenate([channels[:dim_L], np.tile([2], dim_Z)])
            dim_L = np.count_nonzero(channels_cur==0)
            dim_R = np.count_nonzero(channels_cur==1)
            layers.append(SplitChannels(dim, channels=channels_cur))
        elif ltype == '>':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke merge layer.')
            channels_cur = np.concatenate([channels[:(dim_L+dim_R)], np.tile([2], dim_Z)])
            dim_L += dim_R
            dim_R = 0
            layers.append(MergeChannels(dim, channels=channels_cur))
        elif ltype == 'R':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke Real NVP layer.')
            S1 = nonlinear_transform(dim_R, nlayers=nl_layers_scale, nhidden=nl_hidden_scale,
                                     activation=nl_activation_scale, init_outputs=1.5, **layer_args)
            T1 = nonlinear_transform(dim_R, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, init_outputs=1.5, **layer_args)
            S2 = nonlinear_transform(dim_L, nlayers=nl_layers_scale, nhidden=nl_hidden_scale,
                                     activation=nl_activation_scale, init_outputs=1.5, **layer_args)
            T2 = nonlinear_transform(dim_L, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, init_outputs=1.5, **layer_args)
            layers.append(RealNVP([S1, T1, S2, T2]))
        elif ltype == 'r':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke RealNVP layer.')
            layers.append(RealNVP([S1, T1, S2, T2]))
        elif ltype == 'W':
            if dim_R > 0:
                raise RuntimeError('Not merged. Cannot invoke whitening layer.')
            W = FixedWhiten(whiten, keepdims=whiten_keepdims)
            dim_L = W.keepdims
            dim_Z = dim-W.keepdims
            layers.append(W)

    if energy_model is None:
        return InvNet(dim, layers, prior=prior)
    else:
        return EnergyInvNet(energy_model, layers, prior=prior)
