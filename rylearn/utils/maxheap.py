"""Provides an easy MaxHeap interface.

Author: Ryan Strauss
"""

import heapq


class MaxHeap:
    """A thin wrapper class that implements a MaxHeap.

    See Also:
        `heapq` for documentation on the each function's corresponding
            MinHeap behavior.
    """

    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for item in self._data:
            yield item

    def push(self, element):
        heapq.heappush(self._data, -element)

    def pop(self):
        return -heapq.heappop(self._data)

    def replace(self, element):
        return -heapq.heapreplace(self._data, -element)

    def peek(self):
        return -self._data[0]

    def pushpop(self, element):
        return -heapq.heappushpop(self._data, -element)

    def nlargest(self, n):
        return [-e for e in heapq.nsmallest(n, self._data)]
