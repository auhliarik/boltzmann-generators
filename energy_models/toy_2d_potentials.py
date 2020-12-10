import numpy as np
import tensorflow as tf


class DoubleWell:
    default_params = {
        'dim': 2,
        'a': 4.0,
        'b': 12.0,
        'c': 1.0,
        'd': 1.0
    }

    def __init__(self, params=None):
        # Set parameters
        if params is None:
            params = self.__class__.default_params
        self.params = params
        self.dim = self.params['dim']

    def energy(self, x):
        # dimer_energy: x-part of energy which has landscape of a dimer
        # oscillator_energy: y-part of energy which is basically a LHO
        dimer_energy = 0.25 * self.params['a'] * x[:, 0] ** 4 \
                       - 0.5 * self.params['b'] * x[:, 0] ** 2 \
                       + self.params['c'] * x[:, 0]
        oscillator_energy = 0.5 * self.params['d'] * x[:, 1] ** 2
        return dimer_energy + oscillator_energy

    def energy_tf(self, x):
        dimer_energy = 0.25 * self.params['a'] * x[:, 0] ** 4 \
                       - 0.5 * self.params['b'] * x[:, 0] ** 2 \
                       + self.params['c'] * x[:, 0]
        oscillator_energy = 0.5 * self.params['d'] * x[:, 1] ** 2
        return dimer_energy + oscillator_energy

    def plot_dimer_energy(self, axis=None, temperature=1.0):
        """ Plots the dimer energy to the standard figure """
        x_grid = np.linspace(-3, 3, num=200)
        X = np.hstack([x_grid[:, None], np.zeros((x_grid.size, self.dim - 1))])
        energies = self.energy(X) / temperature

        import matplotlib.pyplot as plt
        if axis is None:
            axis = plt.gca()
        #plt.figure(figsize=(5, 4))
        axis.plot(x_grid, energies, linewidth=3, color='black')
        axis.set_xlabel('x / a.u.')
        axis.set_ylabel('Energy / kT')
        axis.set_ylim(energies.min() - 2.0, energies[int(energies.size / 2)] + 2.0)

        return x_grid, energies


class MuellerPotential:
    default_params = {
        'dim': 2,
        'alpha': 1.0,
        'a': [-1, -1, -6.5, 0.7],
        'b': [0, 0, 11, 0.6],
        'c': [-10, -10, -6.5, 0.7],
        'A': [-200, -100, -170, 15],
        'X': [1, 0, -0.5, -1],
        'Y': [0, 0.5, 1.5, 1]
    }

    def __init__(self, params=None):
        # Set parameters
        if params is None:
            params = self.__class__.default_params
        self.params = params
        self.dim = self.params['dim']

    def energy(self, x):
        """Muller potential

        Returns
        -------
        potential : {float, np.ndarray}
            Potential energy. Will be the same shape as the inputs, x and y.

        Reference
        ---------
        Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
        """
        params = self.params    # shorthand
        x1 = x[:, 0]
        x2 = x[:, 1]
        value = 0
        for i in range(4):
            value += params['A'][i] * np.exp(
                params['a'][i] * (x1 - params['X'][i]) ** 2
                + params['b'][i] * (x1 - params['X'][i]) * (x2 - params['Y'][i])
                + params['c'][i] * (x2 - params['Y'][i]) ** 2
            )

        return self.params['alpha'] * value

    def energy_tf(self, x):
        """Muller potential

        Returns
        -------
        potential : {float, np.ndarray}
            Potential energy. Will be the same shape as the inputs, x and y.

        Reference
        ---------
        Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
        """
        params = self.params  # shorthand
        x1 = x[:, 0]
        x2 = x[:, 1]
        batch_size = tf.shape(x)[0]
        value = tf.zeros(batch_size)
        for i in range(4):
            value += params['A'][i] * tf.exp(
                params['a'][i] * (x1 - params['X'][i]) ** 2
                + params['b'][i] * (x1 - params['X'][i]) * (x2 - params['Y'][i])
                + params['c'][i] * (x2 - params['Y'][i]) ** 2
            )

        return self.params['alpha'] * value
