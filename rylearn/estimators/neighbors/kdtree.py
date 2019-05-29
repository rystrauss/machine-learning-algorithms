"""Contains an impoementation of the k-d tree data structure.

The k-dimensional tree is a space-partitioning data structure for
organizing points in a k-dimensional space. It is particularly well suited
to nearest neighbor searches.

https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

Author: Ryan Strauss
"""

import heapq

import numpy as np

from ...metrics.pairwise import euclidean_distance


class MaxHeap:

    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def push(self, element):
        heapq.heappush(self._data, -element)

    def pop(self):
        return -heapq.heappop(self._data)

    def replace(self, element):
        return -heapq.heapreplace(self._data, element)

    def peek(self):
        return -self._data[0]


class Node:

    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class KDTree:

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be an ndarray.')
        if data.ndim != 2:
            raise ValueError('data must be a 2-dimensional list of points.')

        self._k = data.shape[1]
        self._root = self._construct(data)

    def _construct(self, data, depth=0):
        axis = depth % self._k
        data = np.sort(data, axis=0, order=[axis], kind='mergesort')
        median = len(data) // 2

        left_child = self._construct(data[:median], depth + 1)
        right_child = self._construct(data[median + 1:], depth + 1)

        return Node(data[median], left_child, right_child)

    def nearest_neighbors(self, point, k=1, distance=euclidean_distance):
        best = MaxHeap()
        self._nearest_neightbors(self._root, best, point, k, distance)

    def _nearest_neightbors(self, node, best, point, k, distance):
        pass
