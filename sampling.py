import numpy as np
import pyemma

from analysis import bar


class MetropolisGauss:
    """ Metropolis Monte-Carlo simulation with Gaussian proposal steps """

    def __init__(self, model, x0, temperature=1.0, sigma_metro=0.1,
                 burn_in=0, stride=1, n_walkers=1, mapper=None):
        """
        Arguments:
            model:
                Energy model object, must have the method energy(x).
            x0 (array):
                Initial configuration.
            sigma_metro (float):
                Standard deviation of Gaussian proposal step.
            temperature (float or array):
                Temperature. By default (1.0) the energy is interpreted in reduced units.
                When given an array, its length must correspond to n_walkers, then the walkers
                are simulated at different temperatures.
            burn_in (int):
                Number of burn-in steps that will not be saved.
            stride (int):
                Every so many-th step will be saved.
            n_walkers (int):
                Number of parallel walkers.
            mapper:
                Object with method map(X), which for example removes permutation.
                If given will be applied to each accepted configuration.
        """
        self.model = model
        self.sigma_metro = sigma_metro
        self.temperature = temperature
        self.burn_in = burn_in
        self.stride = stride
        self.n_walkers = n_walkers

        # Initiate random generator
        self.random_gen = np.random.default_rng()

        # Couple of additional initializations
        self.x = None
        self.energy = None
        self.step = 0
        self.trajectory_ = []
        self.energy_trajectory_ = []

        # Assign mapper and set sampler to initial position
        if mapper is None:
            class DummyMapper:
                @staticmethod
                def map(X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper
        self.reset(x0)

    def _proposal_step(self):
        # Produce a proposal step
        self.x_prop = self.x + self.random_gen.normal(0, self.sigma_metro, self.x.shape)
        self.x_prop = self.mapper.map(self.x_prop)
        self.E_prop = self.model.energy(self.x_prop)

    def _acceptance_step(self):
        # Acceptance step
        acc = -np.log(np.random.random_sample()) > ((self.E_prop - self.energy) / self.temperature)
        self.x = np.where(acc, self.x_prop, self.x)
        self.energy = np.where(acc, self.E_prop, self.energy)

    def reset(self, x0):
        # Counters
        self.step = 0
        self.trajectory_ = []
        self.energy_trajectory_ = []

        # Prepare initial configuration
        self.x = np.tile(x0, (self.n_walkers, 1))
        self.x = self.mapper.map(self.x)
        self.energy = self.model.energy(self.x)

        # Save the first frame if no burn-in period
        if self.burn_in == 0:
            self.trajectory_.append(self.x)
            self.energy_trajectory_.append(self.energy)

    def change_energy_model(self, model):
        """ Changes energy model on the fly to the new one """
        self.model = model

    @property
    def trajectories(self):
        """ Returns a list of trajectories, one trajectory for each walker """
        t = np.array(self.trajectory_).astype(np.float32)
        return [t[:, i, :] for i in range(t.shape[1])]

    @property
    def trajectory(self):
        return self.trajectories[0]

    @property
    def energy_trajectories(self):
        """ Returns a list of energy trajectory, one trajectory for each walker """
        e = np.array(self.energy_trajectory_)
        return [e[:, i] for i in range(e.shape[1])]

    @property
    def energy_trajectory(self):
        return self.energy_trajectories[0]

    def run(self, n_steps=1, verbose=0):
        for i in range(n_steps):
            self._proposal_step()
            self._acceptance_step()
            self.step += 1
            if verbose and i % verbose == 0:
                print('Step', i, '/', n_steps)
            if self.step > self.burn_in and self.step % self.stride == 0:
                self.trajectory_.append(self.x)
                self.energy_trajectory_.append(self.energy)


class UmbrellaModel:
    def __init__(self, energy_model, rc_function, k_umbrella, m_umbrella):
        """ Umbrella Energy Model

        Wraps given energy model U(x) and returns energy E(x) = U(x) + k*(rc(x) - m)**2

        Arguments:
            energy_model :
                Unbiased energy model object that provides function energy(x).
            k_umbrella (float):
                Force constant of umbrella potential.
            m_umbrella (float):
                Mean position of RC in umbrella potential.
            rc_function (function):
                Function to compute reaction coordinate value.
        """
        self.energy_model = energy_model
        if energy_model is not None:
            self.dim = energy_model.dim
        self.rc_function = rc_function
        self.k_umbrella = k_umbrella
        self.m_umbrella = m_umbrella
        self.rc_trajectory = None

    def bias_energy(self, rc):
        return self.k_umbrella * (rc - self.m_umbrella)**2

    def energy(self, x):
        rc = self.rc_function(x)
        return self.energy_model.energy(x) + self.bias_energy(rc)


class UmbrellaSampling:
    def __init__(self, energy_model, sampler, rc_function, x0,
                 n_umbrella, k, m_min, m_max, forward_backward=True):
        """ Umbrella Sampling

        Arguments:
            energy_model:
                Energy model object that provides function energy(x).
            sampler:
                Sampler - object with methods reset(x), run(n_steps)
                and change_energy_model(model).
            rc_function (function):
                Function to compute reaction coordinate value.
            x0 (np.ndarray):
                Initial configuration for the sampler. Will be used only in the first
                window. Initial configuration in other windows will be the last configuration
                in the previous window.
            n_umbrella (int):
                Number of umbrella windows (simulations) in forward run.
            k (float):
                Force constant for umbrella potential.
            m_min (float):
                Mean value of reaction coordinate in the first window.
            m_max (float):
                Mean value of reaction coordinate in the last window.
            forward_backward (bool):
                If True, umbrella simulation is run both forwards and backwards.
        """
        self.energy_model = energy_model
        self.sampler = sampler
        self.rc_function = rc_function
        self.x0 = x0
        self.forward_backward = forward_backward

        diff = (m_max - m_min) / (n_umbrella - 1)
        m_umbrella = [m_min + i*diff for i in range(n_umbrella)]
        if forward_backward:
            m_umbrella += reversed(m_umbrella)
        self.umbrellas = [UmbrellaModel(energy_model, rc_function, k, m) for m in m_umbrella]

    def run(self, n_steps=10000, verbose=True):
        x_start = self.x0
        for i in range(len(self.umbrellas)):
            if verbose:
                print('Umbrella', i+1, '/', len(self.umbrellas))
            self.sampler.change_energy_model(self.umbrellas[i])
            self.sampler.reset(x_start)
            self.sampler.run(n_steps=n_steps)
            trajectory = self.sampler.trajectory
            rc_trajectory = self.rc_function(trajectory)
            self.umbrellas[i].rc_trajectory = rc_trajectory
            x_start = np.array([trajectory[-1]])

    @property
    def rc_trajectories(self):
        return [u.rc_trajectory for u in self.umbrellas]

    @property
    def bias_energies(self):
        return [u.bias_energy(u.rc_trajectory) for u in self.umbrellas]

    @property
    def umbrella_positions(self):
        return np.array([u.m_umbrella for u in self.umbrellas])

    def umbrella_free_energies(self):
        free_energies = [0]
        for i in range(len(self.umbrellas) - 1):
            k_umbrella = self.umbrellas[i].k_umbrella
            # Free energy difference between two consecutive umbrellas
            # Ua calculated for samples from A
            ua_sampled_in_a = (
                k_umbrella 
                * (self.umbrellas[i].rc_trajectory - self.umbrellas[i].m_umbrella)**2
            )
            ub_sampled_in_a = (
                k_umbrella
                * (self.umbrellas[i].rc_trajectory - self.umbrellas[i+1].m_umbrella)**2
            )
            ua_sampled_in_b = (
                k_umbrella
                * (self.umbrellas[i+1].rc_trajectory - self.umbrellas[i].m_umbrella)**2
            )
            ub_sampled_in_b = (
                k_umbrella
                * (self.umbrellas[i+1].rc_trajectory - self.umbrellas[i+1].m_umbrella)**2
            )
            delta_F = bar(ub_sampled_in_a - ua_sampled_in_a, ua_sampled_in_b - ub_sampled_in_b)
            free_energies.append(free_energies[-1] + delta_F)
        return np.array(free_energies)

    def mbar(self, rc_min=None, rc_max=None, rc_bins=50):
        """ Estimates free energy along reaction coordinate with MBAR.

        Arguments:
            rc_min (float or None):
                Minimum bin position. If None, the minimum RC value will be used.
            rc_max (float or None):
                Maximum bin position. If None, the maximum RC value will be used.
            rc_bins (int):
                Number of bins

        Returns:
            tuple:
                bins (np.ndarray):
                    Bin positions (their centers).
                free_energy (np.ndarray):
                    Free energy, i.e. -log(p), for all bins.
        """
        if rc_min is None:
            rc_min = np.concatenate(self.rc_trajectories).min(initial=0)
        if rc_max is None:
            rc_max = np.concatenate(self.rc_trajectories).max(initial=0)
        x_grid = np.linspace(rc_min, rc_max, rc_bins)

        # List of RC trajectories, one array for each UmbrellaModel.
        rc_trajectories = [rc_trajectory.astype(np.float64) for rc_trajectory in self.rc_trajectories]
        # Assign number of bin in x_grid for RC values.
        digitized_categories = [np.digitize(rc_trajectory, x_grid) for rc_trajectory in self.rc_trajectories]
        umbrella_centers = [u.m_umbrella for u in self.umbrellas]
        umbrella_force_constants = [2.0*u.k_umbrella for u in self.umbrellas]

        mbar_obj = pyemma.thermo.estimate_umbrella_sampling(
            rc_trajectories, digitized_categories,
            umbrella_centers, umbrella_force_constants,
            estimator='mbar')

        x_grid_mean = np.concatenate([x_grid, [2*x_grid[-1] - x_grid[-2]]])
        x_grid_mean -= 0.5*(x_grid[1]-x_grid[0])

        F = np.zeros(x_grid_mean.size)
        F[mbar_obj.active_set] = mbar_obj.stationary_distribution
        F = -np.log(F)

        return x_grid_mean, F
