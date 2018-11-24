import numpy as np


def mse(x1, x2):
    assert x1.shape == x2.shape, 'x1 and x2 must have the same shape'
    assert x1.ndim == 1, 'x1 and x2 must be one dimensional'

    return np.sum(np.square(x1 - x2)) / x1.shape[0]


def mae(x1, x2):
    assert x1.shape == x2.shape, 'x1 and x2 must have the same shape'
    assert x1.ndim == 1, 'x1 and x2 must be one dimensional'

    return np.sum(np.abs(x1 - x2)) / x1.shape[0]
