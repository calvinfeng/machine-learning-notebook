import numpy as np
import pdb


def sgd(w, dw, config=None):
    """Performs vanilla stochastic gradient descent.
    """
    if config is None:
        config = {}
    
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    
    return w, config


def adam(x, dx, config=None):
    """Uses Adam update rule, which incorporates moving averages of both the gradient and its square
    and a bias correction term.

    Available config:
    - learning_rate: Scalar learning rate. 
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving avarge of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    
    config.setdefault('learning_rate', 1e-1)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dx**2
    config['t'] += 1

    alpha = config['learning_rate'] * np.sqrt(1 - config['beta2']**config['t']) / (1 - config['beta1']**config['t'])
    x -= alpha * (config['m'] / (np.sqrt(config['v']) + config['epsilon']))

    return x, config