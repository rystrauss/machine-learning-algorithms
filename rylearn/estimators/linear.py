"""This module provides implementations of linear models.

Author: Ryan Strauss
"""
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from .base import BaseEstimator
from ..activations import sigmoid
from ..data import batch
from ..metrics import r_squared, accuracy_score

_VALID_PENALTIES = ['l2', 'l1']


class SGDEstimator(BaseEstimator, ABC):
    """Base class for stochastic gradient descent (SGD) models."""

    def __init__(self,
                 penalty='l2',
                 alpha=1e-4,
                 lr=1e-5,
                 fit_intercept=True,
                 epochs=1000,
                 shuffle=True):
        """Constructor.

        Args:
            penalty (str, None): The penalty (aka regularization term) to
            be used. Defaults to 'l2'.
            alpha (float): Constant that multiplies the regularization
            term. Defaults to 1e-4.
            lr (float): The initial learning rate. Defaults to 1e-5.
            fit_intercept (bool): Whether or not to fit the intercept
            term. Defaults to True.
            epochs (int): Number of passes over the training data.
            Defaults to 1000.
            shuffle (bool): Whether or not the training data should be
            shuffled after each epoch. Defaults to True.
        """
        if penalty is not None and penalty not in _VALID_PENALTIES:
            raise ValueError(
                'Penalty should be one of {}.'.format(_VALID_PENALTIES))

        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.shuffle = shuffle

        self.theta = None

    @abstractmethod
    def _compute_gradients(self, batch_x, batch_y, batch_size):
        """Computes the gradients for the given batch.

        Args:
            batch_x: The minibatch features.
            batch_y: The minibatch targets.
            batch_size: The minibatch size.

        Returns:
            The gradients for the examples in the provided minibatch.
        """
        pass

    def _pre_fit(self, x, y, batch_size):
        """This method is run at the beginning of the `fit` method. Can
        be implemented to allow for additional checks and pre-fitting setup."""
        pass

    def fit(self, x, y, batch_size=32):
        """Fits the model to the provided training data.

        Args:
            x: A 2-dimentional matrix containing the training features.
            y: A vector that contains the target values.
            batch_size: Minibatch size for stochastic gradient descent.

        Returns:
            None.
        """
        if not isinstance(x, (np.ndarray, list)):
            raise TypeError(
                'X must be of type list or ndarray, but found {}'.format(
                    type(x)))
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError(
                'y must be of type list or ndarray, but found {}'.format(
                    type(y)))

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self._pre_fit(x, y, batch_size)

        # Add a bias term
        if self.fit_intercept:
            x = np.insert(x, 0, 1, axis=1)

        # Initialize the weights
        if self.theta is None:
            self.theta = np.random.uniform(size=(x.shape[1],))

        # Perform gradient descent
        iteration = 0
        for _ in tqdm(range(self.epochs), desc='Epochs'):
            # Shuffle the data before each epoch
            if self.shuffle:
                indices = np.random.permutation(len(x))
                x = x[indices]
                y = y[indices]

            for batch_x, batch_y in batch(x, y, batch_size=batch_size):
                gradients = self._compute_gradients(
                    batch_x, batch_y, batch_size)

                if self.penalty is not None:
                    penalty = 0
                    if self.penalty == 'l2':
                        penalty = 2 * self.theta[1:]
                    elif self.penalty == 'l1':
                        penalty = 1

                    penalty /= batch_size

                    if self.fit_intercept:
                        self.theta[0] -= self.lr * gradients[0]
                        self.theta[1:] -= \
                            self.lr * gradients[1:] + self.alpha * penalty
                    else:
                        self.theta -= self.lr * gradients + self.alpha * penalty
                else:
                    self.theta -= self.lr * gradients

            iteration += 1


class SGDRegressor(SGDEstimator):
    """Linear regression model optimized with Stochastic Gradient Descent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_gradients(self, batch_x, batch_y, batch_size):
        preds = batch_x.dot(self.theta)
        error = preds - batch_y
        gradients = batch_x.T.dot(error) / batch_size

        return gradients

    def predict(self, x):
        if self.fit_intercept:
            x = np.insert(x, 0, 1, axis=1)
        return x.dot(self.theta)

    def score(self, x, y):
        """Evaluates the model's performance on labeled data by determining the
        R^2 value between the labels and the predicitons.

        Args:
            x: A 2-dimentional matrix containing the features.
            y: A vector that contains the target values.

        Returns:
            The R^2 value between the ground truth labels and the
            model's predicitons.
        """
        return r_squared(y, self.predict(x))


class SGDClassifier(SGDEstimator):
    """Logistic regression model optimized with Stochastic Gradient Descent.

    This implementation only works with binary classification problems.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_gradients(self, batch_x, batch_y, batch_size):
        preds = sigmoid(batch_x.dot(self.theta))
        error = preds - batch_y
        gradients = batch_x.T.dot(error) / batch_size

        return gradients

    def _pre_fit(self, x, y, batch_size):
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('Target values must either be 0 or 1.')

    def predict(self, x):
        if self.fit_intercept:
            x = np.insert(x, 0, 1, axis=1)
        return sigmoid(x.dot(self.theta))

    def predict_class(self, x, threshold=0.5):
        """Makes class predictions, rather than returning the raw
        logistic output.

        Args:
            x: A 2-dimentional matrix containing the examples to make
            predictions on.
            threshold: The threshold for determining which class to
            assign to an example. Must be in the range [0, 1].

        Returns:
            The predicted class labels.
        """
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be in the range 0 to 1.')

        if self.fit_intercept:
            x = np.insert(x, 0, 1, axis=1)
        preds = sigmoid(x.dot(self.theta))
        return np.where(preds > threshold, 1, 0).astype(np.uint8)

    def score(self, x, y, threshold=0.5):
        """Evaluates the model's performance on labeled data by determining the
        classification accuracy.

        Args:
            x: A 2-dimentional matrix containing the features.
            y: A vector that contains the target values.
            threshold: The threshold for determining which class to
            assign to an example. Must be in the range [0, 1].

        Returns:
            The model's classification accuracy.
        """
        return accuracy_score(y, self.predict_class(x, threshold=threshold))
