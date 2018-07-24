# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np


def sgd(w, dw, config=None):
    """Performs vanilla stochastic gradient descent

    Args:
        w: matrix weights
        dw: gradients of weights
        config:
            - learning_rate: scalar learning rate
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw

    return w, config


def sgd_momentum(w, dw, config=None):
    """Performs stochastic gradient descent with momentum

    Args:
        w: matrix weights
        dw: gradients of weights
        config:
            - learning_rate: scalar learning rate
            - momentum: scalar between 0 and 1, giving the momentum value
            - velocity: a numpy array of the same shape as w and dw, used to store a moving average of the gradients
    """
    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)

    v = config.get('velocity', np.zeros_like(w))
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v

    config['velocity'] = v
    return next_w, config


def rmsprop(x, dx, config=None):
    """Performs RMSProp update rule, which uses a moving average of squared gradient values to set adaptive per-parameter
    learning rates

    Args:
        config:
            - learning_rate: scalar learning rate
            - decay_rate: scalar between 0 and 1, giving the decay rate for the squared gradient cache
            - epsilon: small scalar used for smothing to avoid dividing by zero
            - cache: moving average of second moments of gradients
    """
    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    grad_squared = config['cache']
    grad_squared = config['decay_rate'] * grad_squared + (1 - config['decay_rate']) * dx * dx
    next_x = x - config['learning_rate'] * dx / (np.sqrt(grad_squared) + config['epsilon'])

    config['cache'] = grad_squared
    return next_x, config


def adam(x, dx, config=None):
    """Performs Adam update rule, which incorporates moving average of both the gradient and its square, and a bias
    correction term

    Args:
        config:
            - learning_rate: scalar learning rate
            - beta1: decay rate for moving average of first moment of gradient
            - beta2: decay rate for moving average of second moment of gradient
            - epsilon: small scalar used for smoothing to avoid dividing by zero
            - m: moving average of gradient
            - v: moving average of squared gradient
            - t: iteration number
    """

    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    first_moment = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    sec_moment = config['beta2'] * config['v'] + (1 - config['beta2']) * dx * dx
    next_x = x - config['learning_rate'] * first_moment / (np.sqrt(sec_moment) + config['epsilon'])

    config['m'] = first_moment
    config['v'] = sec_moment
    return next_x, config
