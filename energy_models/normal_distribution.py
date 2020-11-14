import numpy as np
import tensorflow as tf


class NormalDistribution:
    def __init__(self, dim, sigma=1):
        self.dim = dim
        self.sigma = sigma

    def produce_train_dataset(self, num_samples):
        return np.random.normal(0, self.sigma, size=(num_samples, self.dim))

    def energy(self, x):
        return (0.5 / self.sigma**2) * np.sum(x**2)

    def energy_tf(self, x_pred):
        return (0.5 / self.sigma**2) * tf.math.reduce_sum(x_pred**2, axis=1)

