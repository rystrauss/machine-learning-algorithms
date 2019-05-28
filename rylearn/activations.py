"""Defines common activation functions.

Author: Ryan Strauss
"""

import numpy as np


def sigmoid(x):
    """Passes the elements of the input array through the sigmoid function.

    Args:
        x: The data to pass through the sigmoid function.

    Returns:
        The result of applying the sigmoid function to the input data.
    """
    return 1 / (1 + np.exp(-x))
