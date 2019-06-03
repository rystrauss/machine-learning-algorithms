"""Contains an implementation of the k-d tree data structure.

The k-dimensional tree is a space-partitioning data structure for
organizing points in a k-dimensional space. It is particularly well suited
to nearest neighbor searches.

https://en.wikipedia.org/wiki/K-d_tree

Author: Ryan Strauss
"""

import numpy as np

from rylearn.metrics.distance import squared_distance
from rylearn.utils.maxheap import MaxHeap


class _Node:
    """A node to be used in a k-d tree."""
    __slots__ = ['data', 'right', 'left', 'distance', 'axis', 'label']

    def __init__(self,
                 data=None,
                 left=None,
                 right=None,
                 distance=np.inf,
                 axis=0,
                 label=None):
        self.data = data
        self.left = left
        self.right = right
        self.distance = distance
        self.axis = axis
        self.label = label

    def __lt__(self, other):
        if not isinstance(other, _Node):
            raise ValueError(
                'trying to compare Node to type {}'.format(type(other)))
        return self.distance < other.distance

    def __neg__(self):
        return _Node(data=self.data,
                     left=self.left,
                     right=self.right,
                     distance=-self.distance,
                     label=self.label)


class KDTree:
    """A k-dimensional tree data structure.

    Supports tree construction (obviously) and k-nearest neighbor search.
    """

    def __init__(self, data, labels):
        """Builds the k-d tree from the provided data.

        Args:
            data: A matrix of points to be put into the tree.
            labels: Corresponding labels for the data points.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be an ndarray.')
        if data.ndim != 2:
            raise ValueError('data must be a 2-dimensional list of points.')

        self._k = data.shape[1]
        self._root = self._construct(data, labels)

    def _construct(self, data, targets, depth=0):
        axis = depth % self._k

        if data.size <= 0:
            return None

        indices = data[:, axis].argsort()
        data = data[indices]
        targets = targets[indices]
        median = len(data) // 2

        left_child = self._construct(data[:median], targets[:median], depth + 1)
        right_child = self._construct(data[median + 1:], targets[median + 1:],
                                      depth + 1)

        return _Node(data[median], left_child, right_child, axis=axis,
                     label=targets[median])

    def nearest_neighbors(self,
                          point,
                          k=1,
                          targets_only=True):
        """Finds the k-nearest neighbors in the tree to a given point.

        Args:
            point: The point for which to find the nearest neighbors.
            k: The number of nearest neighbors to return.
            targets_only: If True, on the target values for the neighbors will
                be returned. Otherwise tuples of the form (point, target) will
                be returned.

        Returns:
            The provided point's nearest neighbors.
        """
        if k < 1:
            raise ValueError('k must be at least 1.')

        heap = MaxHeap()
        self._nearest_neighbors(self._root, heap, point, k)

        best = list(heap)

        if targets_only:
            return np.array([node.label for node in sorted(best)])

        return [(node.data, node.label) for node in sorted(best)]

    def _nearest_neighbors(self, node, heap, point, k):
        if node is None:
            return

        node.distance = squared_distance(point, node.data)
        if len(heap) >= k:
            if node.distance < heap.peek().distance:
                heap.replace(node)
        else:
            heap.push(node)

        split_plane = node.data[node.axis]
        plane_dist = squared_distance(point[node.axis], split_plane)

        if point[node.axis] < split_plane:
            self._nearest_neighbors(node.left, heap, point, k)
        else:
            self._nearest_neighbors(node.right, heap, point, k)

        if plane_dist ** 2 < heap.peek().distance or len(heap) < k:
            if point[node.axis] < node.data[node.axis]:
                self._nearest_neighbors(node.right, heap, point, k)
            else:
                self._nearest_neighbors(node.left, heap, point, k)
