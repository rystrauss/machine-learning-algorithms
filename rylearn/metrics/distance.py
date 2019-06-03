"""Definitions of distance metrics.

Author: Ryan Strauss
"""

import numpy as np


def minkowski_distance(x, y, p):
    """Defines the Minkowski distance.

    This is a generalization of both the Euclidean distance and
    the Manhattan distance.

    Args:
        x: A numpy array.
        y: A numpy array.
        p: An integer denoting the order term in the Minkowski distance.

    Returns:
        The Minkowski distance between x and y.
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError('x and y must be numpy arrays.')

    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')

    return np.power(np.sum(np.power((x - y), p)), 1 / p)


def euclidean_distance(x, y):
    """Calculates Euclidian distance.

    Args:
        x: A numpy array.
        y: A numpy array.

    Returns:
        The Euclidean distance between x and y.
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError('x and y must be numpy arrays.')

    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')

    return np.sqrt(np.sum(np.square(x - y)))


def manhattan_distance(x, y):
    """Calculates Manhattan distance.

    Args:
        x: A numpy array.
        y: A numpy array.

    Returns:
        The Manhattan distance between x and y.
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError('x and y must be numpy arrays.')

    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')

    return np.sum(np.abs(x - y))


def squared_distance(x, y):
    """Calculates squared distance. This is the same as Euclidean distance
    squared. Is useful for avoid the square root computation.

    Args:
        x: A numpy array.
        y: A numpy array.

    Returns:
        The squared distance between x and y.
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError('x and y must be numpy arrays.')

    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')

    return np.sum(np.square(x - y))
