from .sgd import SGD

_OPTIMIZERS = {
    'sgd': SGD
}


def get_optimizer(optimizer):
    assert optimizer in _OPTIMIZERS, '{} is not a valid optimizer'.format(optimizer)

    return _OPTIMIZERS[optimizer]
