# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np


def rel_error(x, y):
    """Returns relative error
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    fx = f(x) # Evaluate function value at original point
    grad = np.zeros_like(x)
    itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not itr.finished:
        ix = itr.multi_index
        oldval = x[ix]

        # Evaluate function at x + h, i.e. f(x + h) a.k.a. fxph
        x[ix] = oldval + h
        fxph = f(x)

        # Evaluate function at x - h, i.e. f(x - h) a.k.a. fxmh
        x[ix] = oldval - h
        fxmh = f(x)

        x[ix] = oldval

        # Compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        itr.iternext()

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """Evaluate a numeric gradient for a function that accepts a numpy array and returns a numpy array
    """
    grad = np.zeros_like(x)
    itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not itr.finished:
        ix = itr.multi_index
        oldval = x[ix]

        x[ix] = oldval + h
        pos = f(x).copy()

        x[ix] = oldval - h
        neg = f(x).copy()

        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        itr.iternext()

    return grad
