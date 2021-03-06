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

    def __init__(self, layers, energy_model, **args_for_layers):
        """ Initiate BG and create its layers if necessary.

        Arguments:
            layers (list or string):
                List of invertible layers OR a string containing layer marks, which will be used
                to produce invertible layers by function create_layers_for_boltzmann_generator.
                In the first case, layers must have implemented methods 'connect_xz', 'connect_zx'
                used when connecting layers to form TF models and methods 'log_det_xz',
                'log_det_zx' if log(det(J)) of their transformations is not 0.
            energy_model:
                Instance of a class having:
                - attribute 'dim' (physical dimension of the system to be sampled by BG)
                - method 'energy' (calculates per sample energy on batch, returns np.ndarray)
                - method 'energy_tf' (same as above, returns tf.Tensor)
        """
        for attribute in ("dim", "energy", "energy_tf"):
            if not hasattr(energy_model, attribute):
                raise AttributeError(f"Provided energy does not have attribute '{attribute}'")

        self.dim = energy_model.dim

        if isinstance(layers, str):
            layers = create_layers_for_boltzmann_generator(
                self.dim, layer_types=layers, **args_for_layers
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

        self.layers = layers
        self.energy_model = energy_model
        # Create models from individual transformations
        self.connect_layers()
        # Other attributes
        self.optimizer = None

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
                self.log_det_xz = tf.keras.layers.Lambda(lambda x: x, name=log_det_name)(
                    log_det_xzs[0]
                )
            else:
                self.log_det_xz = tf.keras.layers.Add(name=log_det_name)(log_det_xzs)
            self.FxzJ = tf.keras.Model(
                inputs=self.input_x,
                outputs=[self.output_z, self.log_det_xz],
                name="FxzJ"
            )

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
                self.log_det_zx = tf.keras.layers.Lambda(lambda x: x, name=log_det_name)(
                    log_det_zxs[0]
                )
            else:
                self.log_det_zx = tf.keras.layers.Add(name=log_det_name)(log_det_zxs)
            self.FzxJ = tf.keras.Model(
                inputs=self.input_z,
                outputs=[self.output_x, self.log_det_zx],
                name="FzxJ"
            )

    def transform_xz(self, x):
        """ Transforms batch or single configuration in X to Z """
        return self.Fxz.predict(util.ensure_shape(x))

    def transform_zx(self, z):
        """ Transforms batch or single configuration in Z to X """
        return self.Fzx.predict(util.ensure_shape(z))

    def transform_xz_with_jacobian(self, x):
        """ Maps batch in X by Fxz and returns batch in Z as well as log(det(Jxz)) """
        x = util.ensure_shape(x)
        if self.FxzJ is None:
            return self.Fxz.predict(x), np.zeros(x.shape[0])
        else:
            z, log_jacobian = self.FxzJ.predict(x)
            return z, log_jacobian[:]

    def transform_zx_with_jacobian(self, z):
        """ Maps batch in Z by Fzx and returns batch in X as well as log(det(Jzx)) """
        z = util.ensure_shape(z)
        if self.FxzJ is None:
            return self.Fzx.predict(z), np.zeros(z.shape[0])
        else:
            x, log_jacobian = self.FzxJ.predict(z)
            return x, log_jacobian[:]

    def train(self, x, x_val=None, batch_size=1024, iterations=2000, lr=0.001,
              weight_ML=0, weight_KL=0,
              verbose=True, print_training_info_interval=10, print_total_loss_only=False,
              optimizer=None, clipnorm=None,
              high_energy=1e6, max_energy=1e10,
              temperature=1.0,
              weight_RCEnt=0.0, rc_function=None, rc_min=0.0, rc_max=1.0,
              return_validation_energies=False):
        """ Train the Boltzmann generator (i.e. its X <-> Z transformations).

        Arguments:
            x (np.ndarray):
                Training set - 2D numpy array of shape (training_set_size, bg.dim).
            x_val (np.ndarray od None):
                Validation dataset. If None, validation is disabled.
            batch_size (int):
                Size of the batch to be used while training.
            iterations (int):
                Number of training iterations that should be executed.
                Note that this is NOT an epoch - one iteration means one training step
                (i.e. params update) using single batch.
            lr (float):
                Learning rate used by optimizer in training.
                Not used if instance of optimizer is given in 'optimizer'.
            weight_ML (float):
                Weight of the ML (maximum-likelihood) loss - training by example.
            weight_KL (float):
                Weight of the KL (Kullback-Leibler divergence) loss - training by energy.
            verbose (bool):
                If true, training info is periodically printed.
            print_training_info_interval (int):
                If in verbose mode, training info is printed every (this param)-th iteration.
            print_total_loss_only (bool):
                If true, losses are summed in the info print.
                Otherwise their individual values are printed.
            optimizer (str or tf.keras.optimizers.Optimizer or None):
                Optimizer to be used in training. By default uses the same optimizer instance
                as the last time / creates new instance of Adam optimizer if running for
                the first time. If set to 'reset', newly created Adam optimizer is used.
            clipnorm (float or None):
                Clip norm (clipping of gradient by norm) used by optimizer.
                Not used if instance of optimizer is given in 'optimizer'.
            high_energy (float):
                E_high (start of logarithm) used by linlogcut when training by energy.
            max_energy (float):
                E_max (energy cutoff) used by linlogcut when training by energy.
            temperature (float):
                kT - real temperature (NOT relative, which is used in other methods) of the
                physical systems. Has to be in the same unit of energy as used by
                energy_model.energy and energy_model.energy_tf methods.
            weight_RCEnt (float):
                Weight of the RC-entropy loss.
            rc_function (function or None):
                Function that takes batch as an input and returns 1D array of
                RC (reaction coordinate) values (i.e. RC per sample).
            rc_min (float):
                Minimal value of the RC.
                Note: Do not use too low (high) values of rc_min (rc_max) for good
                functionality of RC-entropy loss in training.
            rc_max (float):
                Maximal value of the RC.
            return_validation_energies (bool):
                If true, arrays with energies of sampled X and Z-configurations is returned
                as well (both with shape (iterations, batch_size)).

        Returns:
            tuple:
                loss_values (dict):
                    Dictionary, where keys are loss names and corresponding values are
                    np.ndarray-s with loss values in individual iterations.
                energies_x_val (list):
                    2D list with energies of X-samples from validation set in all iterations.
                    Batch size for validation is the same as for training.
                energies_z_val (list):
                    The same as above for Z-samples.
        """

        used_ML_loss = False
        used_KL_loss = False
        used_RC_loss = False
        validation_enabled = False if x_val is None else True

        inputs = []
        outputs = []
        applied_losses = []

        # List of tuples (loss_layer, loss_name) which will be used as metrics.
        # Otherwise train_on_batch would return only sum of losses as we use
        # model.add_loss method to set losses.
        losses_for_metrics = []

        # Dummy true values. Although this is an unsupervised learning, TF throws error
        # if no y (targets) are provided. Therefore, arrays of zeros are used as targets
        # for all outputs.
        y = []

        # Choose model inputs and outputs according to used losses.
        if weight_ML > 0:
            used_ML_loss = True
            inputs.append(self.input_x)

            outputs.append(self.output_z)
            outputs.append(self.log_det_xz)

            y.append(np.zeros((batch_size, self.dim)))
            y.append(np.zeros(batch_size))

            ml_loss = tf.keras.layers.Lambda(losses.LossMLNormal(weight_ML), name="ML_loss_layer")(
                [self.output_z, self.log_det_xz]
            )
            applied_losses.append(ml_loss)
            losses_for_metrics.append((ml_loss, "ML_loss"))

        if weight_KL > 0:
            used_KL_loss = True
            inputs.append(self.input_z)

            outputs.append(self.output_x)
            outputs.append(self.log_det_zx)

            y.append(np.zeros((batch_size, self.dim)))
            y.append(np.zeros(batch_size))

            instance_of_kl_loss_class = losses.LossKL(
                weight_KL, self.energy_model.energy_tf, high_energy, max_energy, temperature
            )
            kl_loss = tf.keras.layers.Lambda(instance_of_kl_loss_class, name="KL_loss_layer")(
                [self.output_x, self.log_det_zx]
            )
            applied_losses.append(kl_loss)
            losses_for_metrics.append((kl_loss, "KL_loss"))

        if weight_RCEnt > 0:
            used_RC_loss = True
            if not used_KL_loss:
                inputs.append(self.input_z)
                outputs.append(self.output_x)
                y.append(np.zeros((batch_size, self.dim)))

            rc_loss = tf.keras.layers.Lambda(losses.LossRCEntropy(weight_RCEnt, rc_function, rc_min, rc_max))(
                self.output_x
            )
            applied_losses.append(rc_loss)
            losses_for_metrics.append((rc_loss, "RC_loss"))

        # Build optimizer
        if optimizer is None and self.optimizer:
            optimizer = self.optimizer
            optimizer.lr.assign(lr)
            optimizer.clipnorm = clipnorm
        elif optimizer is None or optimizer == "reset":
            optimizer = tf.keras.optimizers.Adam(lr=lr, clipnorm=clipnorm)
        elif isinstance(optimizer, str):
            raise Exception(
                "Parameter 'optimizer' can be only one of "
                "None, 'reset' or optimizer instance."
            )
        self.optimizer = optimizer

        # Construct model
        dual_model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="dual_model")
        dual_model.add_loss(applied_losses)
        for loss, loss_name in losses_for_metrics:
            dual_model.add_metric(loss, name=loss_name)
        dual_model.compile(optimizer=optimizer)

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
        # Create numpy arrays during at the beginning and then
        # overwrite (which is faster then appending).
        loss_values = {"loss": np.zeros(iterations)}
        if validation_enabled:
            loss_values["val_loss"] = np.zeros(iterations)
        for _, loss_name in losses_for_metrics:
            loss_values[loss_name] = np.zeros(iterations)
            if validation_enabled:
                loss_values["val_" + loss_name] = np.zeros(iterations)
        energies_x_val = []
        energies_z_val = []

        # Training loop
        for iteration in range(iterations):
            # Sample the batch
            input_for_training = []

            if used_ML_loss:
                x_batch = x[np.random.choice(training_set_indices, size=batch_size, replace=True)]
                input_for_training.append(x_batch)
            if used_KL_loss or used_RC_loss:
                z_batch = np.random.standard_normal(size=(batch_size, self.dim))
                input_for_training.append(z_batch)

            # Train model on single batch. Losses are returned in dictionary.
            losses_for_this_iteration = dual_model.train_on_batch(
                x=input_for_training, y=y, return_dict=True
            )

            # Save loss values
            for loss_name in losses_for_this_iteration:
                loss_values[loss_name][iteration] = losses_for_this_iteration[loss_name]

            # Validation
            if validation_enabled:
                input_for_validation = []
                z_val_batch = None
                x_val_batch = None

                if used_ML_loss:
                    x_val_batch = x[np.random.choice(validation_set_indices, size=batch_size, replace=True)]
                    input_for_validation.append(x_val_batch)
                if used_KL_loss or used_RC_loss:
                    z_val_batch = np.random.standard_normal(size=(batch_size, self.dim))
                    input_for_validation.append(z_val_batch)

                val_losses_for_this_iteration = dual_model.test_on_batch(
                    x=input_for_validation, y=y, return_dict=True
                )

                # Save validation loss values
                for loss_name in val_losses_for_this_iteration:
                    loss_values["val_" + loss_name][iteration] = val_losses_for_this_iteration[loss_name]

                # Return energies of output x and z configurations from validation batch
                if return_validation_energies:
                    if z_val_batch is not None:
                        x_out = self.Fzx(z_val_batch)
                        energies_x_val.append(self.energy_model.energy(x_out))
                    if x_val_batch is not None:
                        z_out = self.Fxz(x_val_batch)
                        energies_z_val.append(self.energy_z(z_out))

            # Print training info
            if verbose:
                if iteration % print_training_info_interval == 0:
                    i = iteration
                    info_str = f"Iteration {i}/{iterations}: "
                    for type_of_loss in loss_values:
                        if print_total_loss_only and type_of_loss not in ("loss", "val_loss"):
                            continue
                        info_str += f"{type_of_loss}: {loss_values[type_of_loss][i]:.2f} "

                    print(info_str)
                    sys.stdout.flush()

        if return_validation_energies:
            return loss_values, energies_x_val, energies_z_val
        else:
            return loss_values

    def energy_z(self, z, relative_temperature=1.0):
        """ Returns energies of given z-configurations """
        energies = (
            self.dim * np.log(np.sqrt(relative_temperature))
            + np.sum(z ** 2 / (2 * relative_temperature), axis=1)
        )
        return energies

    def sample_z(self, n_sample, relative_temperature=1.0, return_energies=False):
        """ Samples from prior distribution in Z.

        Arguments:
            relative_temperature (float):
                Relative temperature (tau in the article). Equal to the
                variance of the isotropic Gaussian sampled in z-space.
            n_sample (int):
                Number of samples to be returned.
            return_energies (bool):
                If True, function returns both samples from z-space and z-energies
                of these configurations. Otherwise (which is default) return only the samples.

        Returns:
            tuple:
                sample_z (np.ndarray):
                    Array of samples in z-space.
                energy_z (np.ndarray):
                    Energies of z samples.
        """
        # Random samples from N(mu, sigma^2) can be obtained from standard normal distribution
        # N(0,1) by sampling: mu + sigma * np.random.standard_normal(size=...)
        sample_z = np.sqrt(relative_temperature) * np.random.standard_normal(size=(n_sample, self.dim))

        if return_energies:
            energies = self.energy_z(sample_z)
            return sample_z, energies
        else:
            return sample_z

    def sample(self, n_sample, relative_temperature=1.0):
        """ Samples from prior distribution in Z and returns them together with
        transformed X configurations and other relevant quantities.

        Arguments:
            relative_temperature (float):
                Relative relative_temperature (tau in the article). Equal to the variance of
                the isotropic Gaussian sampled in z-space. Defaults to 1.
            n_sample (int):
                Number of samples to be returned.

        Returns:
            tuple:
                sample_z (np.ndarray):
                    Samples in z-space.
                sample_x (np.ndarray):
                    Samples in x-space obtained by mapping z samples.
                energy_z (np.ndarray):
                    Energies of z samples in z space.
                energy_x (np.ndarray):
                    Energies of x samples in x (real) space.
                log_w (np.ndarray):
                    Logarithm of statistical weights of samples (see article).
        """
        # Get sample in Z space
        sample_z, energy_z = self.sample_z(
            n_sample=n_sample, relative_temperature=relative_temperature, return_energies=True
        )
        # Transform this sample to X sample (together with jacobian of transformation)
        sample_x, log_jacobian_of_transform_zx = self.transform_zx_with_jacobian(sample_z)

        energy_x = self.energy_model.energy(sample_x) / relative_temperature
        log_weights = -energy_x + energy_z + log_jacobian_of_transform_zx

        return sample_z, sample_x, energy_z, energy_x, log_weights

    def save(self, dir_and_prefix):
        """ Saves model weights and optimizer state """
        # It is enough to save just one model model with jacobian, as all
        # four models (Fxz, Fzx, FxzJ, FzxJ) share their weights.
        if not self.optimizer:
            print("BG does not have optimizer (probably it has been not trained yet)."
                  "Saving default optimizer.")
            self.optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(model=self.FzxJ, optimizer=self.optimizer)
        checkpoint.save(dir_and_prefix)

    def load(self, dir_and_prefix, load_latest_checkpoint=False):
        """ Loads model weights and optimizer state.
        Make sure that you are loading data into BG with the same parameters.

        Arguments:
            dir_and_prefix (str):
               Path to the directory + prefix of checkpoint to be loaded
               (including "-{id}" i.e. for instance "resources/bg_model-1").
            load_latest_checkpoint (bool):
                If true, only name of the directory should be passed to first argument
                and latest checkpoint is found and loaded.
        """
        if load_latest_checkpoint:
            dir_and_prefix = tf.train.latest_checkpoint(dir_and_prefix)
            if not dir_and_prefix:
                raise Exception(f"Could not find any TF checkpoint in directory '{dir_and_prefix}'")
        if not self.optimizer:
            self.optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(model=self.FzxJ, optimizer=self.optimizer)
        load_status = checkpoint.restore(dir_and_prefix)
        load_status.assert_existing_objects_matched()


def create_layers_for_boltzmann_generator(
        dim, layer_types,
        channels=None,
        nl_layers=2, nl_hidden=100, nl_activation='relu',
        nl_layers_scale=None, nl_hidden_scale=None, nl_activation_scale='tanh',
        whiten=None,  # whiten_keepdims=None,
        **layer_args):
    """ Returns list of layers for Boltzmann generator.
    Mainly designed to be used in its constructor.

    Arguments:
        dim:
            Dimension of the (physical) system
        layer_types:
            String describing the sequence of layers. Usage:
                R RealNVP layer
                r RealNVP layer that shares parameters with the previous layer
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
            String; hidden-neuron activation function used in the layers
            of nonlinear transformations.
        nl_layers_scale:
            Number of hidden layers in the scaling networks
            (denoted as S1, S2,... in the article). If None, nl_layers is used.
        nl_hidden_scale:
            Number of hidden units in each layer of scaling networks
            (nonlinear transformations). If None, nl_hidden is used.
        nl_activation_scale:
            String; hidden-neuron activation functions used in scaling networks.
            If None, nl_activation will be used.
        whiten:
            Can be None or array. If not None, computes a whitening
            transformation with respect to given coordinates.
            Currently not implemented.
    """
    # whiten_keepdims:
    #     Can be None or int. Number of largest-variance dimensions
    #     to keep after whitening.
    #     Currently not implemented.

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
