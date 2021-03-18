import numpy as np
import tensorflow as tf

from typing import Union

from util import distance_matrix_squared


class LennardJonesCluster:
    """ Energy model for system of 7 LJ-interacting particles in a cluster. """
    default_params = {
        'n_particles': 7,
        'epsilon': 10.0,        # LJ energy factor
        'sigma': 1,             # LJ particle size
        'origin_k': 15.0,       # force constant of CoM binding to origin
    }

    def __init__(self, params: dict = None):
        # Set parameters.
        if not params:
            params = self.__class__.default_params
        self.params = params
        self.dim = params["n_particles"] * 2

        # Create mask matrix to help computing particle interactions.
        n_particles = self.params["n_particles"]
        self.mask_matrix = np.ones((n_particles, n_particles), dtype=np.float32)
        # Particles do not interact with themselves.
        for i in range(n_particles):
            self.mask_matrix[i, i] = 0.0

    def init_positions(self, scaling_factor=1.12) -> np.ndarray:
        """ Initializes particle positions

        Returns array of particle coordinates. One particle is placed
        in the origin and other are placed periodically in a circle around
        the center one.

        Arguments:
            scaling_factor (float):
                Scaling factor to be applied to the configuration.
                Namely the distance between center particle and particles
                in the circle given in multiples of sigma.
        """
        positions = list()
        # Particle in the center.
        positions.append(np.array([0., 0.]))

        # Other particles are placed periodically
        # clockwise in a circle around the center particle.
        n_particles_except_center = self.params["n_particles"] - 1
        angle_diff = 2 * np.pi / n_particles_except_center
        for i in range(n_particles_except_center):
            angle = -np.pi - i*angle_diff
            x = self.params["sigma"] * np.cos(angle)
            y = self.params["sigma"] * np.sin(angle)
            positions.append(np.array([x, y]))

        return scaling_factor * np.array(positions).reshape((1, 2*self.params["n_particles"]))

    def permute_particles(self, x, new_labels):
        """ Return configuration where particle have changed labels (IDs)

        Arguments:
            x (np.ndarray):
                Initial configuration.
            new_labels (list):
                List of integers with new labels for the particles, which are numbered 1,...,N.
                For example for 3 particles [2, 1, 3] means that particles 1 and 2 will have their
                labels swapped in the new configuration.
        Returns:
            np.ndarray:
                New configuration with labels permuted.
        """
        n_particles = self.params["n_particles"]
        x_original = x.reshape((n_particles, 2))
        x_new = np.ones_like(x_original)

        for i, label in enumerate(new_labels):
            # Labels start from 1, not 0.
            label -= 1
            x_new[i] = x_original[label]

        return x_new.reshape((1, 2*n_particles))

    def draw_config(
            self, x, axis=None, box_size=8, fig_size=5,
            with_numbers=False, particle_colors=None, alpha=0.7):
        """ Draw given cluster configuration

        Arguments:
            x (np.ndarray):
                Configuration of particles.
            axis (matplotlib.axis.Axis):
                Axis where image should be drawn.
                By default creates new figure with axis.
            fig_size (float):
                Size (width/height) of the figure in inches.
            box_size (float):
                Dimension of visible square box around particles. In multiples of sigma.
            with_numbers (bool):
                If True, particle numbers are displayed.
            particle_colors (dict):
                Dictionary where keys are numbers of particles and values are their colors.
                Particles are numbered 1,...,N and by default they are grey.
                For example: {1: "red", 2:"blue"}.
                Color of marker particle.
            alpha (float):
                Alpha of particle circles.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Prepare data.
        x = x.reshape(((self.params["n_particles"]), 2))
        if not particle_colors:
            particle_colors = dict()
        # Set up figure.
        if axis is None:
            plt.figure(figsize=(fig_size, fig_size))
            axis = plt.gca()

        d = box_size / 2
        axis.set_xlim((-d, d))
        axis.set_ylim((-d, d))
        axis.set_xticks([])
        axis.set_yticks([])

        # Draw solvent particles.
        for i, circle_position in enumerate(x):
            # We want circle indices to start from 1.
            i += 1
            color = particle_colors.get(i, "grey")

            axis.add_patch(
                Circle(circle_position, radius=0.5*self.params["sigma"], linewidth=2,
                       edgecolor="black", facecolor=color, alpha=alpha)
            )
            if with_numbers:
                axis.text(
                    *circle_position, i, fontsize=14,
                    horizontalalignment='center', verticalalignment='center',
                )

        return axis

    def LJ_energy(self, x: np.ndarray) -> np.ndarray:
        """ Calculates energy of Lennard-Jones interaction between particles """
        # All component-wise distances between particles.
        batch_size = np.shape(x)[0]
        distances_squared = distance_matrix_squared(x, x, dim=2)

        mask_matrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batch_size, 1, 1))
        # This is just to avoid NaNs (distance of particle from itself is 0),
        # the inverses will be set to 0 after multiplication by mask_matrix.
        distances_squared = distances_squared + (1.0 - mask_matrix)
        rel_distances_squared = (self.params["sigma"]**2) / distances_squared
        # Remove self-interactions.
        rel_distances_squared = rel_distances_squared * mask_matrix

        # Calculate energy
        rel_distances_power_6 = rel_distances_squared**3
        energy = (
            np.sum(rel_distances_power_6**2, axis=(1, 2))
            - np.sum(rel_distances_power_6, axis=(1, 2))
        )
        # Add factor 1/2 because we have double-counted each interaction.
        energy *= 0.5 * 4 * self.params["epsilon"]
        return energy

    def origin_constraint_energy(self, x: np.ndarray) -> np.ndarray:
        """ Calculates energy of potential, that attracts center of mass to origin """
        energy_x = 0.5 * self.params["origin_k"] * np.mean(x[:, 0::2])**2
        energy_y = 0.5 * self.params["origin_k"] * np.mean(x[:, 1::2])**2
        return energy_x + energy_y

    def energy(self, x: np.ndarray) -> np.ndarray:
        """ Calculates energy of configuration x, given as numpy array """
        return self.LJ_energy(x) + self.origin_constraint_energy(x)

    def LJ_energy_tf(self, x: tf.Tensor) -> tf.Tensor:
        """ Calculates energy of Lennard-Jones interaction between particles """
        x_comp = x[:, 0::2]
        y_comp = x[:, 1::2]
        batch_size = tf.shape(x)[0]
        n_particles = tf.shape(x_comp)[1]

        # All component-wise distances between particles.
        # Following variables contain lists of square matrices, one for each configuration
        # in the batch. Each matrix has one row for each particle and each of these rows
        # contain copied x/y coordinate of this particle.
        x_comp_tiled = tf.tile(tf.expand_dims(x_comp, 2), [1, 1, n_particles])
        y_comp_tiled = tf.tile(tf.expand_dims(y_comp, 2), [1, 1, n_particles])
        distances_x = x_comp_tiled - tf.transpose(x_comp_tiled, perm=[0, 2, 1])
        distances_y = y_comp_tiled - tf.transpose(y_comp_tiled, perm=[0, 2, 1])
        distances_squared = distances_x**2 + distances_y**2

        mask_matrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batch_size, 1, 1])
        # This is just to avoid NaNs (distance of particle from itself is 0),
        # the inverses will be set to 0 after multiplication by mask_matrix.
        distances_squared = distances_squared + (1.0 - mask_matrix)
        rel_distances_squared = (self.params["sigma"]**2) / distances_squared
        # Remove self-interactions.
        rel_distances_squared = rel_distances_squared * mask_matrix

        # Calculate energy.
        rel_distances_power_6 = rel_distances_squared**3
        energy = (
            tf.reduce_sum(rel_distances_power_6**2, axis=(1, 2))
            - tf.reduce_sum(rel_distances_power_6, axis=(1, 2))
        )
        # Add factor 1/2 because we have double-counted each interaction.
        energy *= 0.5 * 4 * self.params["epsilon"]
        return energy

    def origin_constraint_energy_tf(self, x: tf.Tensor) -> tf.Tensor:
        """ Calculates energy of potential, that attracts center of mass to origin """
        energy_x = 0.5 * self.params["origin_k"] * tf.reduce_mean(x[:, 0::2], axis=1)**2
        energy_y = 0.5 * self.params["origin_k"] * tf.reduce_mean(x[:, 1::2], axis=1)**2
        return energy_x + energy_y

    def energy_tf(self, x: tf.Tensor) -> tf.Tensor:
        """ Calculates energy of configuration x, given as tensorflow tensor """
        return self.LJ_energy_tf(x) + self.origin_constraint_energy_tf(x)

    @staticmethod
    def _switch_function(
            r: Union[np.ndarray, tf.Tensor],
            r0: float = 1.3, d0: float = 0.3, n: int = 6) -> Union[np.ndarray, tf.Tensor]:
        """ Switch function for calculation of coordination number

        Works both with np.ndarray and tf.Tensor.
        It's used as a smooth substitution for
        f(r) = 1  if r <= r0
               0  if r > r0

        Uses the following formula:
        s(r) = (1 - x) / (1 - x**2)  where x(r) = ((r - d0) / r0) ** n
        See 'rational' in https://www.plumed.org/doc-v2.5/user-doc/html/switchingfunction.html
        """
        x = ((r - d0) / r0) ** n
        return (1 - x) / (1 - x**2)

    @staticmethod
    def coordination_number(x, i=1, **switch_function_kwargs) -> np.ndarray:
        """ Calculate coordination number of a particle

        Computes distances between particles, processes them with _switch_function,
        sums results and returns 1D array with CNs for each configuration in a batch.

        Arguments:
            x (np.ndarray):
                Batch (matrix) of configurations.
            i (int):
                Number of particle for which CN should be calculated.
                Particle indices start with 1, not 0.
            switch_function_kwargs (dict):
                Keyword arguments for switching function that processes
                particle distances.
        """
        i -= 1
        batch_size = x.shape[0]
        n_particles = x.shape[1] // 2
        # Position of i-th particle.
        x_i = x[:, i*2: (i + 1)*2]
        # Get matrix with one row for each configuration in x
        # and columns containing distances from x_i.
        distances_squared = distance_matrix_squared(x_i, x, dim=2)
        distances = np.sqrt(distances_squared).reshape((batch_size, n_particles))
        # Process distances with switch function.
        distances_switched = LennardJonesCluster._switch_function(
            distances, **switch_function_kwargs
        )
        # Particle does not contribute to its own coordination number.
        distances_switched[:, i] = 0.0
        output = np.sum(distances_switched, axis=1)
        return output

    @staticmethod
    def coordination_number_tf(x, i=1, **switch_function_kwargs) -> tf.Tensor:
        """ Calculate coordination number of a particle

        Computes distances between particles, processes them with _switch_function,
        sums results and returns 1D array with CNs for each configuration in a batch.

        Arguments:
            x (tf.Tensor):
                Batch (matrix) of configurations.
            i (int):
                Number of particle for which CN should be calculated.
                Particle indices start with 1, not 0.
            switch_function_kwargs (dict):
                Keyword arguments for switching function that processes
                particle distances.
        """
        # Particles are numbered from 1.
        i -= 1
        x_comp = x[:, 0::2]
        y_comp = x[:, 1::2]
        n_particles = tf.shape(x_comp)[1]

        # Position of i-th particle - 1D array with value per sample.
        x_i = x[:, i*2]
        y_i = x[:, i*2 + 1]
        # Create matrices with one row for each configuration (sample) that
        # contains copied value of x/y coordinate of i-th particle.
        x_i_tiled = tf.tile(tf.expand_dims(x_i, 1), [1, n_particles])
        y_i_tiled = tf.tile(tf.expand_dims(y_i, 1), [1, n_particles])

        # Calculate distances.
        distances_x = x_comp - x_i_tiled
        distances_y = y_comp - y_i_tiled
        distances = tf.sqrt(distances_x ** 2 + distances_y ** 2)
        # Process distances with switch function.
        distances_switched = LennardJonesCluster._switch_function(
            distances, **switch_function_kwargs
        )

        # Remove i-th column as particle does not contribute
        # to its own coordination number.
        output = tf.reduce_sum(
            tf.concat([distances_switched[:, :i], distances_switched[:, (i + 1):]], axis=1),
            axis=1
        )
        return output
