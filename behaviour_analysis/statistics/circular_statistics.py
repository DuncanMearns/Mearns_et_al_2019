"""Author: Joe Donovan"""
from __future__ import division, print_function
import numpy as np
from scipy.special import i0
from scipy.stats.stats import _validate_distribution


def circ_energy_distance_fast(u_values, v_values):
    """
    Compute the circular energy distance - adapated from the scipy code
    Assumes input in radians
    """
    u_values, u_weights = _validate_distribution(u_values, None)
    v_values, v_weights = _validate_distribution(v_values, None)

    # Modulus the input data
    u_values = u_values % (np.pi * 2)
    v_values = v_values % (np.pi * 2)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Change deltas to circular
    dtemp = deltas % (2 * np.pi)
    dtemp[dtemp > np.pi] = 2 * np.pi - dtemp[dtemp > np.pi]
    deltas = dtemp

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    u_cdf = u_cdf_indices / u_values.size
    v_cdf = v_cdf_indices / v_values.size
    return np.sqrt(2) * np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))


# other bits that might be useful:

def vonmises_kde(data, kappa, n_bins=100):
    """
    Apply a von mises style KDE to handle circular data
    Adapted from from https://stackoverflow.com/questions/28839246/scipy-gaussian-kde-and-circular-data#
    """
    x = np.linspace(0, 2 * np.pi, n_bins)
    # integrate von mises kernels
    kde = np.exp(kappa * np.cos(x[:, None] - data[None, :])).sum(1) / (2 * np.pi * i0(kappa))
    # normalize so the integral is 1
    kde /= np.trapz(kde, x=x)
    return x, kde


def polar_average(theta1, theta2, weight=None):
    """
    Take the (weighted) vector average, since a normal mean doesn't make sense for circular quantities
    Note - theta should be in radians
    see https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    """
    if weight is None:
        weight = np.ones_like(theta1) * .5
    return np.arctan2(np.sin(theta1) * weight + (1 - weight) * np.sin(theta2), np.cos(theta1) * weight + (1 - weight) * np.cos(theta2)) % (2 * np.pi)
