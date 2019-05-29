"""Contains an impoementation of the k-d tree data structure.

The k-dimensional tree is a space-partitioning data structure for
organizing points in a k-dimensional space. It is particularly well suited
to nearest neighbor searches.

https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

Author: Ryan Strauss
"""

import heapq

import numpy as np

from rylearn.metrics.pairwise import squared_distance


class _MaxHeap:

    def __init__(self, max_size=None):
        self.max_size = max_size

        self._data = []

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for item in self._data:
            yield item

    def push(self, element):
        if self.max_size is not None and len(self._data) >= self.max_size:
            heapq.heappushpop(self._data, element)
        else:
            heapq.heappush(self._data, -element)

    def pop(self):
        return -heapq.heappop(self._data)

    def replace(self, element):
        return -heapq.heapreplace(self._data, element)

    def peek(self):
        return -self._data[0]

    def pushpop(self, element):
        return -heapq.heappushpop(self._data, element)


class _Node:

    def __init__(self,
                 data=None,
                 left=None,
                 right=None,
                 distance=np.inf,
                 axis=0,
                 target=None):
        self.data = data
        self.left = left
        self.right = right
        self.distance = distance
        self.axis = axis
        self.target = target

    def __lt__(self, other):
        if not isinstance(other, _Node):
            raise ValueError(
                'trying to compare Node to type {}'.format(type(other)))
        return self.distance < other.distance

    def __neg__(self):
        return _Node(self.data, self.left, self.right, -self.distance)


class KDTree:

    def __init__(self, data, targets):
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be an ndarray.')
        if data.ndim != 2:
            raise ValueError('data must be a 2-dimensional list of points.')

        self._k = data.shape[1]
        self._root = self._construct(data, targets)

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
                     target=targets[median])

    def nearest_neighbors(self, point, k=1, distance_fn=squared_distance):
        best = _MaxHeap(max_size=k)
        self._nearest_neighbors(self._root, best, point, distance_fn)

        return [(node.data, node.target) for node in list(sorted(best))]

    def _nearest_neighbors(self, node, best, point, distance_fn):
        if node is None:
            return

        node.distance = distance_fn(point, node.data)
        best.push(node)

        split_plane = node.data[node.axis]
        plane_dist = np.square(point[node.axis] - split_plane)

        if point[node.axis] < split_plane:
            if node.left is not None:
                self._nearest_neighbors(node.left, best, point, distance_fn)
        elif node.right is not None:
            self._nearest_neighbors(node.right, best, point, distance_fn)

        if -plane_dist > best.peek().distance or len(best) < best.max_size:
            if point[node.axis] < split_plane:
                if node.right is not None:
                    self._nearest_neighbors(node.right, best, point,
                                            distance_fn)
            else:
                if node.left is not None:
                    self._nearest_neighbors(node.left, best, point, distance_fn)


if __name__ == '__main__':
    kdtree = KDTree(np.random.random((500, 10)),
                    np.random.randint(0, 10, (500,)))
    # TODO: figure out why first target is always None
    print(kdtree.nearest_neighbors(np.random.random((10,)), 5)[0])
