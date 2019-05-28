"""Metrics to assess performance on regression tasks.

Author: Ryan Strauss
"""
import numpy as np


def squared_loss(y_true, y_pred):
    return (y_pred - y_true) ** 2


def mse(y_true, y_pred):
    """Mean squared error regression loss.

    Args:
        y_true (ndarray): Ground truth (correct) target values.
        y_pred (ndarray): Estimated target values.

    Returns:
        loss (float): A non-negative floating point value (the best
        value is 0.0), or an array of floating point values, one for each
        individual target.
    """
    return np.average(np.square(y_true - y_pred))


def mae(y_true, y_pred):
    """Mean absolute error regression loss.

        Args:
            y_true (ndarray): Ground truth (correct) target values.
            y_pred (ndarray): Estimated target values.

        Returns:
            loss (float):  A positive floating point value (the best
            value is 0.0).
        """
    return np.average(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    """R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Args:
        y_true (ndarray): Ground truth (correct) target values.
        y_pred (ndarray): Estimated target values.

    Returns:
        z (float): The R^2 score.
    """
    u = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    v = ((y_true - np.mean(y_true)) ** 2).sum(axis=0, dtype=np.float64)
    return 1 - u / v
