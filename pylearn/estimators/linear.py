"""Module supporting linear models.

Author: Ryan Strauss
"""
import warnings

import numpy as np

from ..data import batch
from ..metrics.regression import r_squared


class SGDRegressor:
    """Linear model fitted by minimizing loss with Stochastic Gradient Descent."""

    _VALID_PENALTIES = ['l2', 'l1']

    def __init__(self, penalty='l2', alpha=1e-4, lr=1e-5, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True):
        """
        Args:
            penalty (str): The penalty (aka regularization term) to be used. Defaults to 'l2'.
            alpha (float): Constant that multiplies the regularization term. Defaults to 1e-4.
            lr (float): The initial learning rate. Defaults to 1e-5.
            fit_intercept (bool): Whether or not to fit the intercept term. Defaults to True.
            max_iter (int): Maximum number of loops over the training data (epochs). Defaults to 1000.
            tol (float): The stopping criterion. If it is not None, the iterations will stop when
                         (loss > previous_loss - tol). Defaults to to 1e-3.
            shuffle (bool): Whether or not the training data should be shuffled after each epoch. Defaults to True.
        """
        if penalty is not None and penalty not in self._VALID_PENALTIES:
            raise ValueError('Penalty should be one of {}.'.format(self._VALID_PENALTIES))

        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle

        self.theta = None

    def fit(self, X, y):
        """Fit linear regression model.

        Args:
            X (ndarray, list): Training data.
            y (ndarray, list): Target values.

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
        previous_loss = np.inf
        while True:
            if iteration >= self.max_iter:
                warnings.warn('Maximum number of iterations reached.')
                break

            # Shuffle the data before each epoch
            if self.shuffle:
                indices = np.random.permutation(len(X))
                X = X[indices]
                y = y[indices]

            for batch_X, batch_y in batch(X, y, batch_size=1):
                preds = np.dot(batch_X, self.theta)
                loss = preds - batch_y
                if loss > previous_loss - self.tol:
                    return self

                gradients = batch_X.T.dot(loss)
                if self.fit_intercept and self.penalty is not None:
                    if self.penalty == 'l2':
                        penalty = 2 * self.theta[1:]
                    elif self.penalty == 'l1':
                        indices = self.theta[1:] < 0
                        penalty = np.copy(self.theta[1:])
                        penalty[indices] = np.negative(penalty)[indices]
                    else:
                        penalty = 0

                    self.theta[0] -= self.lr * gradients[0]
                    self.theta[1:] -= self.lr * gradients[1:] + self.alpha * penalty
                else:
                    self.theta -= self.lr * gradients

            iteration += 1

        return self

    def predict(self, X):
        """Predict using the model.

        Args:
            X (ndarray, shape (n_samples, n_features)): Examples to predict on.

        Returns:
            C (ndarray, shape (n_samples,)): Returns predicted values.
        """
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.theta)

    def score(self, X, y):
        """Evaluates the model using the coefficient of determination (r-squared).

        Args:
            X (ndarray): The test data features.
            y (ndarray): The test data targets.

        Returns:
            score (float): Returns the coefficient of determination R^2 of the prediction.
        """
        return r_squared(y, self.predict(X))
