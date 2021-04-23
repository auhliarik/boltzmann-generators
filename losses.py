import tensorflow as tf
import numpy as np
import util


class LossMLNormal:
    """ The maximum-likelihood (ML) loss
    Defined as: U(Fxz(x)) - log(det(Jxz))  where U(z) = (0.5 / prior_sigma**2) * z**2 """

    def __init__(self, weight, prior_sigma=1):
        """
        Arguments:
            weight (float):
                Weight with which the loss should be used. Output of
                __call__ method is multiplied by this factor.
            prior_sigma (float):
                Standard deviation of the isotropic normal prior distribution.
        """
        self.prior_sigma = prior_sigma
        self.weight = weight

    def __call__(self, args) -> tf.Tensor:
        """ Returns array of shape (batch_size,) with calculated values
        of loss for each configuration """
        z_predicted, log_det_jacobian_predicted = args[0], args[1]

        z_energy = (0.5 / (self.prior_sigma**2)) * tf.reduce_sum(z_predicted**2, axis=1)
        return (z_energy - log_det_jacobian_predicted) * self.weight


class LossKL:
    """ The Kullback–Leibler (KL) divergence loss
    Defined as: u(Fzx(z)) - log(det(Jzx))  where u(x) is the reduced energy U(x)/kT """

    def __init__(self, weight, energy_function, high_energy, max_energy, temperature=1.0):
        """
        Arguments:
            energy_function (function):
                Function used to calculate energy of batch of x configurations.
            high_energy (float):
                E_high (start of logarithm) used in linlogcut to prevent overflows.
            max_energy (float):
                E_max (maximal value) used in linlogcut to prevent overflows.
            temperature (float):
                Physical temperature (kT) of the system in the same units
                as used by energy_function. Energy will be devided by this factor.
                Defaults to 1.0, which means that 'energy_function' returns
                the reduced energy.
        """
        self.weight = weight
        self.energy_function = energy_function
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.temperature = temperature  # kT

        self.linlogcut_function = util.linlogcut

    def __call__(self, args) -> tf.Tensor:
        """ Returns array of shape (batch_size,) with calculated values
        of loss for each configuration """
        x_predicted, log_det_jacobian_predicted = args[0], args[1]

        # Compute dimensionless energy
        energy = self.energy_function(x_predicted) / self.temperature
        # Apply linlogcut to prevent overflows
        safe_energy = self.linlogcut_function(energy, self.high_energy, self.max_energy)
        return (safe_energy - log_det_jacobian_predicted) * self.weight


class LossRCEntropy:
    """ Reaction-coordinate (RC) entropy loss
    Defined as: -entropy  where entropy is differential entropy of RC prob. distribution """

    def __init__(self, weight, rc_function, rc_min, rc_max):
        """
        Arguments:
            rc_function (function):
                Function that takes batch as an input and returns 1D array of
                RC (reaction coordinate) values (i.e. RC per sample).
            rc_min (float):
                Minimal value of the RC.
                Note: Do not use too low (high) values of rc_min (rc_max) for good
                functionality of RC-entropy loss in training.
            rc_max (float):
                Maximal value of the RC.
        """
        self.weight = weight
        self.rc_function = rc_function
        self.gauss_means = np.linspace(rc_min, rc_max, 11)
        self.gauss_sigma = (rc_max - rc_min) / 11.0

    def __call__(self, x_predicted) -> tf.Tensor:
        """ Returns array of shape (batch_size,) with calculated values
        of loss for each configuration """
        batch_size = tf.shape(x_predicted)[0]
        # Evaluate RC on batch
        rc = self.rc_function(x_predicted)
        # Change shape from (batch_size,) -> (batch_size, 1)
        rc = tf.expand_dims(rc, axis=1)
        # Create matrix of shape (batch_size, gauss_means), containing
        # values of (not normalized) Gaussian kernel function evaluated at discrete
        # points gauss_means (they can be thought of as means of Gauss functions).
        kernel_matrix = tf.exp(-((rc - self.gauss_means) ** 2) / (2 * self.gauss_sigma**2))
        # Add small number to prevent dividing by zero in the next step.
        kernel_matrix += 1e-6
        # Normalize each row of the matrix (i.e. each kernel).
        kernel_matrix /= tf.reduce_sum(kernel_matrix, axis=1, keepdims=True)
        # Create histogram which is an estimate of RC distribution probability
        # density by calculating mean over kernels (= sum them end divide by their number).
        # Note that this histogram is normalized (sum of bins is 1).
        histogram = tf.reduce_mean(kernel_matrix, axis=0)
        # Calculate entropy of RC distribution
        entropy = -tf.reduce_sum(histogram * tf.math.log(histogram))
        # Ensure that returned tensor has shape (batch_size,)
        return -entropy * self.weight * tf.ones(shape=(batch_size,), dtype=entropy.dtype)
