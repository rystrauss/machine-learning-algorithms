"""Defines base components for all estimators.

Author: Ryan Strauss
"""

from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """Base abstract estimator class that all models must inherit."""

    @abstractmethod
    def fit(self, x, y):
        """Fits the model to the provided training data.

        Args:
            x: A 2-dimentional matrix containing the training features.
            y: A vector that contains the target values.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """Makes predictions with the model.

        Args:
            x: A 2-dimentional matrix containing the examples to make predictions on.

        Returns:
            The predicted targets.
        """
        pass

    @abstractmethod
    def score(self, x, y):
        """Scores the model on the provided data.

        Args:
            x: A 2-dimentional matrix containing the features.
            y: A vector that contains the target values.

        Returns:
            The model's evaluation score (the metric will vary based on the type of model).
        """
        pass
