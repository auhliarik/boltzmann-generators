import tensorflow as tf
import numpy as np
import sys

import losses
import util
from invertible_layers import *
from layers_basic import *


# noinspection PyPep8Naming
class BoltzmannGenerator:
    """ Boltzmann generator (BG) with its basic functionalities. """

    def __init__(self, dim, layers, energy_model=None, **args_for_layers):
        """ Initiate BG and create its layers if necessary.

        Arguments:
            dim:
                Dimension of (physical) system to be sampled by BG.
            layers:
                List of invertible layers OR a string containing
                layer marks, which will be used to produce invertible layers
                by function create_layers_for_boltzmann_generator.
        """
        if isinstance(layers, str):
            layers = create_layers_for_boltzmann_generator(
                dim, layer_types=layers, **args_for_layers
            )

        # Other attributes to be filled later
        # Inputs/outputs
        self.input_x = None
        self.input_z = None
        self.output_x = None
        self.output_z = None
        self.log_det_xz = None
        self.log_det_zx = None

        # Transformations
        self.Fxz = None
        self.Fzx = None
        self.FxzJ = None
        self.FzxJ = None

        self.dim = dim
        self.layers = layers
        self.energy_model = energy_model
        # Create models from individual transformations
        self.connect_layers()

    def connect_xz(self, x):
        z = None
        for i in range(len(self.layers)):
            z = self.layers[i].connect_xz(x)  # connect
            x = z  # rename output to next input
        return z

    def connect_zx(self, z):
        x = None
        for i in reversed(range(len(self.layers))):
            x = self.layers[i].connect_zx(z)  # connect
            z = x  # rename output to next input
        return x

    def connect_layers(self):
        """ Connects layers of the BG both in X->Z and Z->X direction
        and creates 4 TF models: Fxz, Fzx, FxzJ, FzxJ
        Here 'J' means that the model has a second output - logarithm of
        determinant of Jacobian matrix for given input.
        """

        # x -> z
        self.input_x = tf.keras.layers.Input(shape=(self.dim,))
        self.output_z = self.connect_xz(self.input_x)

        # z -> x
        self.input_z = tf.keras.layers.Input(shape=(self.dim,))
        self.output_x = self.connect_zx(self.input_z)

        # Build Fxz and Fzx models
        self.Fxz = tf.keras.Model(inputs=self.input_x, outputs=self.output_z, name="Fxz")
        self.Fzx = tf.keras.Model(inputs=self.input_z, outputs=self.output_x, name="Fzx")

        # Create FxzJ model with calculation of
        # total Jacobian for x->z transformation.
        log_det_xzs = []
        for layer in self.layers:
            if hasattr(layer, 'log_det_xz'):
                log_det_xzs.append(layer.log_det_xz)
        if len(log_det_xzs) == 0:
            self.FxzJ = None
        else:
            log_det_name = "log_det_Fxz"    # name of the layer with log(det(J))
            if len(log_det_xzs) == 1:
                self.log_det_xz = tf.keras.layers.Lambda(lambda x: x, name=log_det_name)(log_det_xzs[0])
            else:
                self.log_det_xz = tf.keras.layers.Add(name=log_det_name)(log_det_xzs)
            self.FxzJ = tf.keras.Model(inputs=self.input_x,
                                       outputs=[self.output_z, self.log_det_xz],
                                       name="FxzJ")

        # Create FzxJ model with calculation of
        # total Jacobian for z->x transformation.
        log_det_zxs = []
        for layer in self.layers:
            if hasattr(layer, 'log_det_zx'):
                log_det_zxs.append(layer.log_det_zx)
        if len(log_det_zxs) == 0:
            self.FzxJ = None
        else:
            log_det_name = "log_det_Fzx"    # name of the layer with log(det(J))
            if len(log_det_zxs) == 1:
                self.log_det_zx = log_det_zxs[0]
                self.log_det_zx = tf.keras.layers.Lambda(lambda x: x, name=log_det_name)(log_det_zxs[0])
            else:
                self.log_det_zx = tf.keras.layers.Add(name=log_det_name)(log_det_zxs)
            self.FzxJ = tf.keras.Model(inputs=self.input_z,
                                       outputs=[self.output_x, self.log_det_zx],
                                       name="FzxJ")

    def transform_xz_with_jacobian(self, x):
        # TODO: Write description
        x = util.ensure_shape(x)
        if self.FxzJ is None:
            return self.Fxz.predict(x), np.zeros(x.shape[0])
        else:
            z, jacobian = self.FxzJ.predict(x)
            return z, jacobian[:]

    def transform_zx_with_jacobian(self, z):
        z = util.ensure_shape(z)
        if self.FxzJ is None:
            return self.Fzx.predict(z), np.zeros(z.shape[0])
        else:
            x, jacobian = self.FzxJ.predict(z)
            return x, jacobian[:]

    def train(self, x, x_val=None, batch_size=1024, iterations=2000, lr=0.001,
              weight_ML=0, weight_KL=0,
              verbose=True, print_training_info_interval=10,
              optimizer=None, clipnorm=None,
              high_energy=1e6, max_energy=1e10, std=1.0,
              temperature=1.0, explore=1.0,
              weight_RCEnt=0.0, rc_func=None, rc_min=0.0, rc_max=1.0,
              return_validation_energies=False):

        used_ML_loss = False
        used_KL_loss = False
        validation_enabled = False if not x_val else True

        inputs = []
        outputs = []
        applied_losses = dict()
        loss_weights = dict()

        # Dummy true values. Although this is an unsupervised learning, TF throws error
        # if no y (targets) are provided. Therefore, arrays of zeros are used as targets
        # for all outputs.
        y = []

        # Names of individual output layers.
        Fxz_output_layer_name = "Fxz_output"
        Fzx_output_layer_name = "Fzx_output"
        log_det_Fxz_output_name = "log_det_Fxz"
        log_det_Fzx_output_name = "log_det_Fzx"

        # Choose model  inputs and outputs according to used losses.
        # Add losses and their weights to dictionaries where keys are names
        # of individual output layers.
        if weight_ML > 0:
            used_ML_loss = True
            inputs.append(self.input_x)

            outputs.append(self.output_z)
            outputs.append(self.log_det_xz)

            y.append(np.zeros((batch_size, self.dim)))
            y.append(np.zeros(batch_size))

            applied_losses[Fxz_output_layer_name] = losses.LossMLNormal()
            applied_losses[log_det_Fxz_output_name] = losses.LossLogDetJacobian()
            loss_weights[Fxz_output_layer_name] = weight_ML
            loss_weights[log_det_Fxz_output_name] = weight_ML

        if weight_KL > 0:
            used_KL_loss = True
            inputs.append(self.input_z)

            outputs.append(self.output_x)
            outputs.append(self.log_det_zx)

            y.append(np.zeros((batch_size, self.dim)))
            y.append(np.zeros(batch_size))

            applied_losses[Fzx_output_layer_name] = losses.LossKL(
                self.energy_model.energy_tf, high_energy, max_energy, temperature
            )
            applied_losses[log_det_Fzx_output_name] = losses.LossLogDetJacobian()
            loss_weights[Fzx_output_layer_name] = weight_KL
            loss_weights[log_det_Fzx_output_name] = weight_KL

        # Old code for RC entropy loss.
        # TODO: implement RC-entropy loss
        # gmeans = None
        # gstd = 0.0
        # if weight_RCEnt > 0.0:
        #     gmeans = np.linspace(rc_min, rc_max, 11)
        #     gstd = (rc_max - rc_min) / 11.0
        # q
        # def loss_RCEnt(y_true, y_pred):
        #     return -self.rc_entropy(rc_func, gmeans, gstd, temperature.size)

        # if weight_RCEnt > 0:
        #     if self.input_z not in inputs:
        #         inputs.append(self.input_z)
        #     if self.output_x not in outputs:
        #     outputs.append(self.output_x)
        # if weight_RCEnt > 0:
        #     if self.input_z not in inputs:
        #         inputs.append(self.input_z)
        #     # if self.output_x not in outputs:
        #     outputs.append(self.output_x)
        #     losses.append(loss_RCEnt)
        #     loss_weights.append(weight_RCEnt)

        # Build optimizer
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(lr=lr, clipnorm=clipnorm)

        # Construct model
        dual_model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="dual_model")
        dual_model.compile(optimizer=optimizer, loss=applied_losses, loss_weights=loss_weights)

        # Data preprocessing
        size_of_training_set = x.shape[0]
        training_set_indices = np.arange(size_of_training_set)
        if x_val is not None:
            size_of_validation_set = x_val.shape[0]
            validation_set_indices = np.arange(size_of_validation_set)
        else:
            validation_set_indices = None

        # Quantities used for monitoring of training.
        # Save loss values in form of numpy arrays in dictionary.
        # Create numpy arrays during the first iteration and then overwrite (faster then appending).
        loss_values = {"loss": np.zeros(iterations)}
        if used_ML_loss:
            loss_values["ML_loss"] = np.zeros(iterations)
        if used_KL_loss:
            loss_values["KL_loss"] = np.zeros(iterations)
        energies_x_val = []
        energies_z_val = []

        # Training loop
        for iteration in range(iterations):
            # Sample the batch
            input_for_training = []

            if used_ML_loss:
                x_batch = x[np.random.choice(training_set_indices, size=batch_size, replace=True)]
                input_for_training.append(x_batch)
            if used_KL_loss:
                z_batch = np.random.standard_normal(size=(batch_size, self.dim))
                input_for_training.append(z_batch)

            # Train model on single batch. Losses are returned in dictionary.
            loss_for_iteration = dual_model.train_on_batch(x=input_for_training, y=y, return_dict=True)

            # Save loss values. Keep in mind that ML and KL losses with physical
            # meaning are implemented using two tensorflow losses.
            loss_values["loss"][iteration] = loss_for_iteration["loss"]
            if used_ML_loss:
                loss_values["ML_loss"][iteration] = loss_for_iteration[Fxz_output_layer_name + "_loss"] \
                                                    + loss_for_iteration[log_det_Fxz_output_name + "_loss"]
            if used_KL_loss:
                loss_values["KL_loss"][iteration] = loss_for_iteration[Fzx_output_layer_name + "_loss"] \
                                                    + loss_for_iteration[log_det_Fzx_output_name + "_loss"]

            # Validate
            # TODO: fully implement validation
            if validation_enabled:
                input_for_validation = []
                z_val_batch = None
                x_val_batch = None

                if used_ML_loss:
                    x_val_batch = x[np.random.choice(validation_set_indices, size=batch_size, replace=True)]
                    input_for_validation.append(x_val_batch)
                if used_KL_loss:
                    z_val_batch = np.random.standard_normal(size=(batch_size, self.dim))
                    input_for_validation.append(z_val_batch)

                val_loss_for_iteration = dual_model.test_on_batch(x=input_for_validation, y=y, return_dict=True)

                if return_validation_energies and x_val_batch and z_val_batch:
                    x_out = self.Fzx(z_val_batch)
                    energies_x_val.append(self.energy_model.energy(x_out))
                    z_out = self.Fxz(x_val_batch)
                    energies_z_val.append(self.energy_z(z_out))

            # Print training info
            if verbose:
                if iteration % print_training_info_interval == 0:
                    i = iteration
                    info_str = f"Iteration {i}/{iterations}: "
                    for type_of_loss in loss_values:
                        info_str += f"{type_of_loss}: {loss_values[type_of_loss][i]:.4f} "

                    print(info_str)
                    sys.stdout.flush()

        if return_validation_energies:
            return loss_values, energies_x_val, energies_z_val
        else:
            return loss_train #, loss_val

        return sample_z, sample_x, #energy_z,    # energy_x, log_weights


def create_boltzmann_generator(
        dim, layer_types, energy_model=None, channels=None,
        nl_layers=2, nl_hidden=100, nl_layers_scale=None, nl_hidden_scale=None,
        nl_activation='relu', nl_activation_scale='tanh',
        whiten=None, whiten_keepdims=None,
        **layer_args):
    """ Returns layers for Boltzmann generator. Mainly used in its constructor.

    Arguments:
    dim:
        Dimension of the (physical) system
    layer_types : str
        String describing the sequence of layers. Usage:
            R RealNVP layer
            r RealNVP layer, share parameters with last layer
        For example: "RRR".
        Splitting and merging layers will be added automatically.
    channels:
        Assignment of dimensions to channels (0/1 array of length dim).
        In majority of use cases you want this to remain None, so that
        default 0,1,0,1... sequence is used.
    nl_layers:
        Number of hidden layers in the nonlinear transformations.
    nl_hidden:
        Number of hidden units in each layer of nonlinear transformations.
    nl_activation:
        String; hidden-neuron activation function used in the layers of nonlinear transformations.
    nl_activation_scale:
        String; hidden-neuron activation functions used in scaling networks.
        If None, nl_activation will be used.
    whiten:
        Can be None or array. If not None, computes a whitening
        transformation with respect to given coordinates.
        Currently not implemented.
    whiten_keepdims:
        Can be None or int. Number of largest-variance dimensions to keep after whitening.
        Currently not implemented.
    """
    # Set up channels
    channels, indices_split, indices_merge = split_merge_indices(dim, n_channels=2, channels=channels)

    # Augment layer types with split and merge layers
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

    # Prepare layers
    layers = []

    # Properties of scale transformations are the same as properties
    # of translational transforms unless they are specified.
    if nl_activation_scale is None:
        nl_activation_scale = nl_activation
    if nl_layers_scale is None:
        nl_layers_scale = nl_layers
    if nl_hidden_scale is None:
        nl_hidden_scale = nl_hidden

    # Number of dimensions left in the signal.
    # The remaining dimensions are going to latent space directly
    dim_left_channel = dim          # number of dimensions in left channel (the default one)
    dim_right_channel = 0           # number of dimensions in right channel
    dim_others = 0                  # currently not used

    # translate and scale layers
    translation_transform_1 = None
    translation_transform_2 = None
    scaling_transform_1 = None
    scaling_transform_2 = None

    for ltype in layer_types:
        print(ltype, dim_left_channel, dim_right_channel, dim_others)
        if ltype == '<':
            if dim_right_channel > 0:
                raise RuntimeError('Already split. Cannot invoke split layer.')
            channels_cur = np.concatenate([channels[:dim_left_channel], np.tile([2], dim_others)])
            dim_left_channel = np.count_nonzero(channels_cur == 0)
            dim_right_channel = np.count_nonzero(channels_cur == 1)
            layers.append(SplitChannels(dim, channels=channels_cur, name_of_merge="Fzx_output"))

        elif ltype == '>':
            if dim_right_channel == 0:
                raise RuntimeError('Not split. Cannot invoke merge layer.')
            channels_cur = np.concatenate([channels[:(dim_left_channel + dim_right_channel)], np.tile([2], dim_others)])
            dim_left_channel += dim_right_channel
            dim_right_channel = 0
            layers.append(MergeChannels(dim, channels=channels_cur, name_of_merge="Fxz_output"))

        elif ltype == 'R':
            if dim_right_channel == 0:
                raise RuntimeError('Not split. Cannot invoke Real NVP layer.')
            scaling_transform_1 = nonlinear_transform(
                dim_right_channel, n_layers=nl_layers_scale, n_hidden=nl_hidden_scale,
                activation=nl_activation_scale, init_outputs=0, **layer_args
            )
            translation_transform_1 = nonlinear_transform(
                dim_right_channel, n_layers=nl_layers, n_hidden=nl_hidden,
                activation=nl_activation, **layer_args
            )
            scaling_transform_2 = nonlinear_transform(
                dim_left_channel, n_layers=nl_layers_scale, n_hidden=nl_hidden_scale,
                activation=nl_activation_scale, init_outputs=0, **layer_args
            )
            translation_transform_2 = nonlinear_transform(
                dim_left_channel, n_layers=nl_layers, n_hidden=nl_hidden,
                activation=nl_activation, **layer_args
            )

            layers.append(RealNVP([scaling_transform_1, translation_transform_1,
                                   scaling_transform_2, translation_transform_2]))

        elif ltype == 'r':
            if dim_right_channel == 0:
                raise RuntimeError('Not split. Cannot invoke RealNVP layer.')
            layers.append(RealNVP([scaling_transform_1, translation_transform_1,
                                   scaling_transform_2, translation_transform_2]))

        elif ltype == 'W':
            raise RuntimeError("Whitening layer not yet implemented")
            # if dim_right_channel > 0:
            #     raise RuntimeError('Not merged. Cannot invoke whitening layer.')
            # W = FixedWhiten(whiten, keepdims=whiten_keepdims)
            # dim_left_channel = W.keepdims
            # dim_others = dim-W.keepdims
            # layers.append(W)

    return layers
