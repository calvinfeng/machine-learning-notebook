import numpy as np


def sgd(w, dw, config={}):
    learning_rate = config.get('learning_rate', 1e-2)
    w -= learning_rate * dw
    return w, config


def adam(x, dx, config={}):
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
    learning_rate = config.get('learning_rate', 1e-3)
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.999)
    epsilon = config.get('epsilon', 1e-8)
    m = config.get('m', np.zeros_like(x))
    v = config.get('v', np.zeros_like(x))
    t = config.get('t', 0)