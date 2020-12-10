import numpy as np


def mean_finite_(x, min_finite=1):
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) > min_finite:
        return np.mean(x[isfin])
    else:
        return np.nan


def std_finite_(x, min_finite=2):

    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) >= min_finite:
        return np.std(x[isfin])
    else:
        return np.nan


def mean_finite(x, axis=None, min_finite=1):
    """ Computes mean over finite values """
    if axis is None:
        return mean_finite_(x, min_finite)
    if axis == 0 or axis == 1:
        M = np.zeros((x.shape[axis-1],))
        for i in range(x.shape[axis-1]):
            if axis == 0:
                M[i] = mean_finite_(x[:, i])
            else:
                M[i] = mean_finite_(x[i])
        return M
    else:
        raise NotImplementedError('axis value not implemented:', axis)


def std_finite(x, axis=None, min_finite=2):
    """ Computes standard deviation over finite values """
    if axis is None:
        return mean_finite_(x, min_finite)
    if axis == 0 or axis == 1:
        S = np.zeros((x.shape[axis-1],))
        for i in range(x.shape[axis-1]):
            if axis == 0:
                S[i] = std_finite_(x[:, i])
            else:
                S[i] = std_finite_(x[i])
        return S
    else:
        raise NotImplementedError('axis value not implemented:', axis)


def free_energy_bootstrap(data, bins=100, bin_range=None, log_weights=None, bias=None, temperature=1.0,
                          n_bootstrap=100, align_bins=None):
    """ Bootstrapped free energy calculation

    If data is a single array, bootstraps by sample.
    If data is a list of arrays, bootstraps by trajectories.

    Arguments:
        data (array or list of arrays):
            Samples in the coordinate in which the free energy will be computed.
        bins (int):
            Number of bins.
        bin_range (None or (float, float)):
            Value range for bins, if not given will be chosen by min and max values of data
        n_bootstrap (int):
            Number of bootstraps to be performed.
        log_weights (None or arrays matching data):
            Sample weights.
        bias (function):
            If not None, the given bias will be removed.
        align_bins (None or indices):
            If not None, will shift samples to align at the given bins indices.

    Returns:
        bin_means (array):
            Array of shape (bins,), mean positions of bins.
        free_energies (array):
            Array of shape (samples, bins), for each bootstrap the free energy of the bins.
    """
    if bin_range is None:
        bin_range = (np.min(data), np.max(data))
    bin_edges = None

    free_energies = []
    by_trajectory = isinstance(data, list)

    for i in np.arange(n_bootstrap):
        selected_indices = np.random.choice(len(data), size=len(data), replace=True)

        if by_trajectory:
            data_sample = np.concatenate([data[i] for i in selected_indices])
            weights_in_sample = None
            if log_weights is not None:
                log_weights_in_sample = np.concatenate([log_weights[i] for i in selected_indices])
                weights_in_sample = np.exp(log_weights_in_sample - log_weights_in_sample.max())

            probability_density_for_sample, bin_edges = np.histogram(
                data_sample, bins=bins, range=bin_range, weights=weights_in_sample, density=True
            )

        else:
            data_sample = data[selected_indices]
            weights_in_sample = None
            if log_weights is not None:
                log_weights_in_sample = log_weights[selected_indices]
                weights_in_sample = np.exp(log_weights_in_sample - log_weights_in_sample.max())

            probability_density_for_sample, bin_edges = np.histogram(
                data_sample, bins=bins, range=bin_range, weights=weights_in_sample, density=True
            )

        free_energy = -np.log(probability_density_for_sample)
        if align_bins is not None:
            free_energy -= free_energy[align_bins].mean()
        free_energies.append(free_energy)
    free_energies = np.vstack(free_energies)
    bin_means = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if bias is not None:
        B = bias(bin_means) / temperature
        free_energies -= B

    return bin_means, free_energies
