"""Definitions of pairwise metrics.

Author: Ryan Strauss
"""

import numpy as np


def minkowski_distance(x, y, p):
    return np.power(np.sum(np.power((x - y), p)), 1 / p)


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))
