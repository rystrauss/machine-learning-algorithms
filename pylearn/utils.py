import numpy as np


def batch(X, y=None, batch_size=32):
    for i in np.arange(0, X.shape[0], batch_size):
        if y is not None:
            yield X[i:i + batch_size], y[i:i + batch_size]
        else:
            yield X[i:i + batch_size]
