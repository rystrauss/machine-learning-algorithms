"""Linear regression models."""
import numpy as np

from ..metrics import mse
from ..optimizers.base import get_optimizer
from ..utils import batch

MAX_ITERATIONS = 1000


class LinearRegressor:

    def __init__(self, alpha, C, fit_intercept=False, epochs=1000, tol=1e-3, optimizer='sgd'):
        self.alpha = alpha
        self.C = C
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.optimizer = get_optimizer(optimizer)
        self.theta = None

    def fit(self, X, y):
        """Fit linear regression model.

        Args:
            X (ndarray, list): Training data
            y (ndarray, list): Target values

        Returns:
            self: Returns an instance of self.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise ValueError('X must be of type list or ndarray, but found {}'.format(type(X)))
        if not isinstance(y, (np.ndarray, list)):
            raise ValueError('y must be of type list or ndarray, but found {}'.format(type(y)))

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Add a bias term
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        # Initialize the weights
        if self.theta is None:
            self.theta = np.random.uniform((X.shape[0],))

        steps = 0
        while steps < self.epochs:
            for batch_X, batch_y in batch(X, y):
                preds = self.predict(X)

        return self

    def predict(self, X):
        """Predict using the model.

        Args:
            X (ndarray, shape (n_samples, n_features)): Samples.

        Returns:
            C (ndarray, shape (n_samples,)): Returns predicted values.
        """
        return np.dot(X, self.theta)
