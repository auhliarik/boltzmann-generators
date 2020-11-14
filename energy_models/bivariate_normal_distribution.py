import numpy as np
import tensorflow as tf


class BivariateNormalDistribution:
    default_params = {
        "sigma_x": 3,
        "sigma_y": 2,
        "rho": 0.9      # correlation between X and Y
    }

    def __init__(self, params=default_params, rg_seed=None):
        self.dim = 2
        self.sigma_x = params["sigma_x"]
        self.sigma_y = params["sigma_y"]
        self.rho = params["rho"]
        print("Used params:\nsigma_x:", self.sigma_x, "sigma_y:", self.sigma_y, "rho:", self.rho, '\n')

        self.mean = np.array([0, 0])
        non_diagonal = self.rho * self.sigma_x * self.sigma_y
        self.cov = np.array([[self.sigma_x**2, non_diagonal],    # covariance matrix
                             [non_diagonal, self.sigma_y**2]])
        print("Covariance matrix of the distribution:", self.cov, sep='\n')

        if rg_seed:     # seed for random generator
            self.rg = np.random.default_rng(rg_seed)
        else:
            self.rg = np.random.default_rng()

    def produce_train_dataset(self, num_samples):
        return self.rg.multivariate_normal(self.mean, self.cov, num_samples, check_valid='raise')

    def energy(self, x):
        """ Computes energy for batch of 2D samples
        x: numpy array or TF tensor of shape (batch_size, 2)
        """
        energy = (x[:, 0]**2) / (self.sigma_x**2) \
                 - 2 * self.rho * x[:, 0] * x[:, 1] / self.sigma_x / self.sigma_y \
                 + (x[:, 1]**2) / (self.sigma_y**2)
        return energy / (2 * (1 - self.rho ** 2))

    energy_tf = energy


