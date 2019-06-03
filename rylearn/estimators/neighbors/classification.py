"""This module provides implementations of nearest neighbor classification
algorithms.

Author: Ryan Strauss
"""

import numpy as np
from scipy.stats import mode

from rylearn.estimators.base import BaseEstimator
from rylearn.estimators.neighbors.kdtree import KDTree
from rylearn.metrics.classification import accuracy_score


class KNeighborsClassifier(BaseEstimator):
    """Implementation of the k-nearest neighbors classification algorithm.

    This implementation uses a k-d tree data structure to store all of the
    points and allow for fast neighbors searches.
    """

    def __init__(self, n_neighbors=5):
        if n_neighbors < 1:
            raise ValueError('k must be at least 1.')

        self.n_neighbors = n_neighbors

        self._kdtree = None

    def fit(self, x, y):
        if not isinstance(x, (np.ndarray, list)):
            raise TypeError(
                'x must be of type list or ndarray, but found {}'.format(
                    type(x)))
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError(
                'y must be of type list or ndarray, but found {}'.format(
                    type(y)))

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self._kdtree = KDTree(x, y)

    def predict(self, x):
        if self._kdtree is None:
            raise RuntimeError('this estimator as not been fitted yet.')

        preds = []
        for point in x:
            nearest = self._kdtree.nearest_neighbors(point, k=self.n_neighbors)
            preds.append(mode(nearest, axis=None)[0])

        return np.array(preds).flatten()

    def score(self, x, y):
        """Evaluates the model's performance on labeled data by determining the
        classification accuracy.

        Args:
            x: A 2-dimentional matrix containing the features.
            y: A vector that contains the target values.

        Returns:
            The model's classification accuracy.
        """
        return accuracy_score(y, self.predict(x))
