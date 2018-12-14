"""Module containing utilites pertaining to data.

Author: Ryan Strauss
"""
import numpy as np


def batch(X, y=None, batch_size=32, repeat=False):
    """
    Separates a set of data into batches.

    Args:
        X (ndarray): The features to be batched.
        y (ndarray): The targets to be batched.
        batch_size (int): Size of the batches.
        repeat (bool): If True, the generator will loop over the dataset indefinitly.

    Returns:
        batches (generator): A generator of data batches of the form (batch_X, batch_y) or (batch_X).
    """
    while True:
        for i in np.arange(0, X.shape[0], batch_size):
            if y is not None:
                yield X[i:i + batch_size], y[i:i + batch_size]
            else:
                yield X[i:i + batch_size]
        if not repeat:
            break
