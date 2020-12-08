import numpy as np


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
