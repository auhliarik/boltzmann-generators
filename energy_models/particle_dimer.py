import numpy as np
import tensorflow as tf

from util import distance_matrix_squared


class ParticleDimer:
    default_params = {
        'n_solvent': 36,
        'eps':  1.0,            # LJ prefactor
        'rm': 1.1,              # LJ particle size
        'dimer_slope': -0.5,    # dimer slope parameter
        'dimer_a': 25.0,        # dimer x2 parameter
        'dimer_b': 10.0,        # dimer x4 parameter
        'dimer_dmid': 1.5,      # dimer transition state distance
        'dimer_k': 20.0,        # dimer force constant
        'box_halfsize': 3.0,
        'box_k': 100.0,         # box repulsion force constant
        'grid_k': 0.0,          # restraint strength to particle grid (to avoid permutation)
    }

    def __init__(self, params=None):
        # Set parameters
        if not params:
            params = self.__class__.default_params
        self.params = params

        # Useful variables
        self.n_particles = params['n_solvent'] + 2
        self.dim = 2 * self.n_particles

        # Create mask matrix to help computing particle interactions
        self.mask_matrix = np.ones((self.n_particles, self.n_particles), dtype=np.float32)
        self.mask_matrix[0, 1] = 0.0
        self.mask_matrix[1, 0] = 0.0
        for i in range(self.n_particles):
            self.mask_matrix[i, i] = 0.0

        # Save grid to compute position restraints
        self.grid = self.init_positions(params['dimer_dmid'])

    def init_positions(self, dimer_distance, scaling_factor=1.05):
        """ Initializes particle positions in a box

        Arguments:
            dimer_distance (float):
                Initial dimer distance.
            scaling_factor (float):
                Scaling factor to be applied to the configuration.
        """
        # Dimer
        pos = list()
        pos.append(np.array([-0.5*dimer_distance, 0]))
        pos.append(np.array([0.5*dimer_distance, 0]))

        # Solvent particles
        sqrt_n = int(np.sqrt(self.params['n_solvent']))
        locations = np.linspace(
            -self.params['box_halfsize']-1, self.params['box_halfsize']+1, sqrt_n+2
        )[1:-1]
        # Solvent particles are arranged in a rectangular mesh
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                pos.append(np.array([locations[i], locations[j]]))
        # Reshape [[x1, y1], [x2, y2]] --> [[x1, y1, x2, y2]]
        pos = np.array(pos).reshape((1, 2*(self.params['n_solvent']+2)))
        return scaling_factor * pos

    def draw_config(self, x, axis=None, dimer_color='blue', alpha=0.7):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        # Prepare data
        X = x.reshape(((self.params['n_solvent']+2), 2))
        # Set up figure
        if axis is None:
            plt.figure(figsize=(5, 5))
            axis = plt.gca()
        d = self.params['box_halfsize']
        axis.set_xlim((-d, d))
        axis.set_ylim((-d, d))

        # Draw box
        border_color = 'lightgrey'
        # Bottom box border (grey zone)
        axis.add_patch(Rectangle(
            (-d-self.params['rm'], -d-self.params['rm']),
            2*d + 2*self.params['rm'],
            0.5*self.params['rm'],
            color=border_color, linewidth=0))
        # Top box border
        axis.add_patch(Rectangle(
            (-d-self.params['rm'], d + 0.5*self.params['rm']),
            2*d + 2*self.params['rm'],
            0.5*self.params['rm'],
            color=border_color, linewidth=0))
        # Left box border
        axis.add_patch(Rectangle(
            (-d-self.params['rm'], -d-self.params['rm']),
            0.5*self.params['rm'],
            2*d + 2*self.params['rm'],
            color=border_color, linewidth=0))
        # Right box border
        axis.add_patch(Rectangle(
            (d + 0.5*self.params['rm'], -d-self.params['rm']),
            0.5*self.params['rm'],
            2*d + 2*self.params['rm'],
            color=border_color, linewidth=0))

        # Draw solvent particles
        circles = []
        for x in X[2:]:
            circles.append(axis.add_patch(
                Circle(x, radius=0.5*self.params['rm'], linewidth=2,
                       edgecolor='black', facecolor='grey', alpha=alpha)))

        # Draw dimer
        circles.append(axis.add_patch(
            Circle(X[0], radius=0.5*self.params['rm'],
                   linewidth=2, edgecolor='black', facecolor=dimer_color, alpha=alpha)))
        circles.append(axis.add_patch(
            Circle(X[1], radius=0.5*self.params['rm'],
                   linewidth=2, edgecolor='black', facecolor=dimer_color, alpha=alpha)))
        # plot(X[:, 0], X[:, 1], linewidth=0, marker='o', color='black')
        
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)
        axis.set_xticks([])
        axis.set_yticks([])
        # return(fig, ax, circles)

    # ANIMATE
    # def animate(i):
    #    X = traj1[i].reshape(n+2, 2)
    #    for i, x in enumerate(X):
    #        circles[i].center = x
    #    return circles

    @staticmethod
    def dimer_distance(x):
        return np.sqrt((x[:, 2] - x[:, 0])**2 + (x[:, 3] - x[:, 1])**2)

    @staticmethod
    def dimer_distance_tf(x):
        return tf.sqrt((x[:, 2] - x[:, 0])**2 + (x[:, 3] - x[:, 1])**2)

    @staticmethod
    def _distance_squared_matrix(crd1, crd2):
        return distance_matrix_squared(crd1, crd2, dim=2)

    def LJ_energy(self, x):
        # All component-wise distances between particles
        batch_size = np.shape(x)[0]
        D2 = distance_matrix_squared(x, x, dim=2)
        mask_matrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batch_size, 1, 1))
        # This is just to avoid NaNs (distance of particle from itself is 0),
        # the inverses will be set to 0 after multiplication by mask_matrix.
        D2 = D2 + (1.0 - mask_matrix)
        D2rel = (self.params['rm']**2) / D2
        # Remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mask_matrix
        # Calculate energy
        # Add factor 1/2 because we have double-counted each interaction
        E = 0.5 * self.params['eps'] * np.sum(D2rel**6, axis=(1, 2))
        return E

    def LJ_energy_tf(self, x):
        # All component-wise distances between particles
        x_comp = x[:, 0::2]
        y_comp = x[:, 1::2]
        batch_size = tf.shape(x)[0]
        n_particles = tf.shape(x_comp)[1]

        X_comp = tf.tile(tf.expand_dims(x_comp, 2), [1, 1, n_particles])
        Y_comp = tf.tile(tf.expand_dims(y_comp, 2), [1, 1, n_particles])
        Dx = X_comp - tf.transpose(X_comp, perm=[0, 2, 1])
        Dy = Y_comp - tf.transpose(Y_comp, perm=[0, 2, 1])
        D2 = Dx**2 + Dy**2

        mask_matrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batch_size, 1, 1])
        # This is just to avoid NaNs (distance of particle from itself is 0),
        # the inverses will be set to 0 after multiplication by mask_matrix.
        D2 = D2 + (1.0 - mask_matrix)
        D2rel = (self.params['rm']**2) / D2
        # Remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mask_matrix
        # Calculate energy
        # Add factor 1/2 because we have double-counted each interaction
        E = 0.5*self.params['eps']*tf.reduce_sum(D2rel**6, axis=(1, 2))
        return E

    def dimer_energy(self, x):
        # Center restraint energy
        energy_dx = self.params['dimer_k'] * (x[:, 0] + x[:, 2])**2
        # y restraint energy
        energy_dy = self.params['dimer_k'] * (x[:, 1])**2 + self.params['dimer_k']*(x[:, 3])**2
        # First two particles
        d = np.sqrt((x[:, 0]-x[:, 2])**2 + (x[:, 1]-x[:, 3])**2)
        d0 = 2 * (d - self.params['dimer_dmid'])
        d2 = d0*d0
        d4 = d2*d2
        energy_interaction = (
            self.params['dimer_slope']*d0
            - self.params['dimer_a']*d2
            + self.params['dimer_b']*d4
        )
        return energy_dx + energy_dy + energy_interaction

    def dimer_energy_tf(self, x):
        # Center restraint energy
        energy_dx = self.params['dimer_k']*(x[:, 0] + x[:, 2])**2
        # y restraint energy
        energy_dy = self.params['dimer_k']*(x[:, 1])**2 + self.params['dimer_k']*(x[:, 3])**2
        # First two particles
        d = tf.sqrt((x[:, 0]-x[:, 2])**2 + (x[:, 1]-x[:, 3])**2)
        d0 = 2 * (d - self.params['dimer_dmid'])
        d2 = d0*d0
        d4 = d2*d2
        energy_interaction = (
            self.params['dimer_slope']*d0
            - self.params['dimer_a']*d2
            + self.params['dimer_b']*d4
        )
        return energy_dx + energy_dy + energy_interaction

    def box_energy(self, x):
        x_comp = x[:, 0::2]
        y_comp = x[:, 1::2]
        # Indicator functions
        E = 0.0
        d_left = -(x_comp + self.params['box_halfsize'])
        E += np.sum((np.sign(d_left) + 1) * self.params['box_k'] * d_left**2, axis=1)
        d_right = (x_comp - self.params['box_halfsize'])
        E += np.sum((np.sign(d_right) + 1) * self.params['box_k'] * d_right**2, axis=1)
        d_down = -(y_comp + self.params['box_halfsize'])
        E += np.sum((np.sign(d_down) + 1) * self.params['box_k'] * d_down**2, axis=1)
        d_up = (y_comp - self.params['box_halfsize'])
        E += np.sum((np.sign(d_up) + 1) * self.params['box_k'] * d_up**2, axis=1)
        return E

    def box_energy_tf(self, x):
        x_comp = x[:, 0::2]
        y_comp = x[:, 1::2]
        # Indicator functions
        E = 0.0
        d_left = -(x_comp + self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_left) + 1) * self.params['box_k'] * d_left**2, axis=1)
        d_right = (x_comp - self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_right) + 1) * self.params['box_k'] * d_right**2, axis=1)
        d_down = -(y_comp + self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_down) + 1) * self.params['box_k'] * d_down**2, axis=1)
        d_up = (y_comp - self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_up) + 1) * self.params['box_k'] * d_up**2, axis=1)
        return E

    def grid_energy(self, x):
        d2 = (x - self.grid)**2
        E = np.sum(self.params['grid_k'] * (self.params['rm']**2 * d2) ** 6, axis=1)
        return E

    def grid_energy_tf(self, x):
        d2 = (x - self.grid)**2
        E = tf.reduce_sum(self.params['grid_k'] * (self.params['rm']**2 * d2) ** 6, axis=1)
        return E

    def _energy(self, x):
        return self.LJ_energy(x) + self.dimer_energy(x) + self.box_energy(x) + self.grid_energy(x)

    def energy(self, x):
        if x.shape[0] < 10000:
            return self._energy(x)
        else:
            energy_x = np.zeros(x.shape[0])
            for i in range(0, len(energy_x), 10000):
                i_from = i
                i_to = min(i_from + 10000, len(energy_x))
                energy_x[i_from:i_to] = self._energy(x[i_from:i_to])
            return energy_x

    def energy_tf(self, x):
        return self.LJ_energy_tf(x) + self.dimer_energy_tf(x) + self.box_energy_tf(x) + self.box_energy_tf(x)

    def plot_dimer_energy(self, axis=None):
        """ Plots the dimer energy to the standard figure """
        x_scan = np.linspace(0.5, 2.5, 100)
        E_scan = self.dimer_energy(np.array([1.5-0.5*x_scan, np.zeros(100), 1.5+0.5*x_scan, np.zeros(100)]).T)
        E_scan -= E_scan.min()

        import matplotlib.pyplot as plt
        if axis is None:
            axis = plt.gca()
        # plt.figure(figsize=(5, 4))
        axis.plot(x_scan, E_scan, linewidth=2)
        axis.set_xlabel('x / a.u.')
        axis.set_ylabel('Energy / kT')
        axis.set_ylim(E_scan.min() - 2.0, E_scan[int(E_scan.size / 2)] + 2.0)

        return x_scan, E_scan
