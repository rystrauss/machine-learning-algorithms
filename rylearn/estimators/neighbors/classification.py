from rylearn.estimators.base import BaseEstimator
from rylearn.estimators.neighbors.kdtree import KDTree


class KNeighborsClassifier(BaseEstimator):

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

        self._kdtree = None

    def fit(self, x, y):
        self._kdtree = KDTree(x, y)

    def predict(self, x):
        pass

    def score(self, x, y):
        pass
