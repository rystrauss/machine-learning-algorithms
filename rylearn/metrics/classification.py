"""Metrics to assess performance on classification tasks.

Author: Ryan Strauss
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels, as returned by a classifier.

    Returns:
        score (float): The fraction of correctly classified samples.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def log_loss(y_true, y_pred):
    """Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in logistic regression, defined as the
    negative log-likelihood of the true labels
    given a probabilistic classifier's predictions. This log loss is only
    defined for two labels.

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted probabilities.

    Returns:
        loss (float)
    """
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
