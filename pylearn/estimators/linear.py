"""
Module supporting linear models.

Author: Ryan Strauss
"""
import warnings

import numpy as np

from ..data import batch
from ..metrics.regression import r_squared


class SGDRegressor:
    """
    Linear model fitted by minimizing loss with Stochastic Gradient Descent.
    """

    def __init__(self, lr=1e-5, fit_intercept=True, max_iter=1000, tol=1e-3):
        """
        Args:
            lr: The learning rate. Defaults to 1e-5.
            fit_intercept: Whether or not to fit the intercept term. Defaults to True.
            max_iter: Maximum number of loops over the training data (epochs). Defaults to 1000.
            tol: The stopping criterion. If it is not None, the iterations will stop when the change is theta (weights)
                 is less than tol. Defaults 1e-3.
        """
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.theta = None

    def fit(self, X, y, batch_size=1):
        """
        Fit linear regression model.

        Args:
            X (ndarray, list): Training data.
            y (ndarray, list): Target values.
            batch_size (int): SGD batch size. Defaults to 1.

        Returns:
            self: Returns an instance of self.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be of type list or ndarray, but found {}'.format(type(X)))
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be of type list or ndarray, but found {}'.format(type(y)))

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Add a bias term
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        # Initialize the weights
        if self.theta is None:
            self.theta = np.random.uniform(size=(X.shape[1],))

        # Perform gradient descent
        iteration = 0
        delta_theta = np.inf
        while True:
            if iteration >= self.max_iter:
                warnings.warn('Maximum number of iterations reached.')
                break

            for batch_X, batch_y in batch(X, y, batch_size=batch_size):
                if delta_theta < self.tol:
                    break

                preds = np.dot(batch_X, self.theta)
                error = preds - batch_y
                gradients = batch_X.T.dot(error) / batch_size

                update = self.theta - self.lr * gradients
                delta_theta = np.absolute(update - self.theta).mean()
                self.theta = update

            iteration += 1

        return self

    def predict(self, X):
        """
        Predict using the model.

        Args:
            X (ndarray, shape (n_samples, n_features)): Examples to predict on.

        Returns:
            C (ndarray, shape (n_samples,)): Returns predicted values.
        """
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.theta)

    def score(self, X, y):
        """
        Evaluates the model using the coefficient of determination (r-squared).

        Args:
            X (ndarray): The test data features.
            y (ndarray): The test data targets.

        Returns:
            score (float): Returns the coefficient of determination R^2 of the prediction.
        """
        return r_squared(y, self.predict(X))
