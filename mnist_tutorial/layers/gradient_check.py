import numpy as np


def rel_error(x, y):
    """Computes relative error
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    """Evaluate numeric gradient for a function that accepts a numpy array and returns a float value.
    This is suited for computing gradients for a function that yields scalar value (e.g. final loss
    function.)
    
    Args:
        f (func): Function to compute gradient on.
        x (numpy.ndarray): An input to the function for which we compute gradient on.
        h (float): Small delta for computing gradient.
    
    Returns:
        grad (numpy.ndarray): Gradient of f with respect to x.
    """
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
    """Evaluate numeric gradient for a function that accepts a numpy array and returns a numpy 
    array. This is suited for computing gradients for a function that yields tensor value (e.g. 
    forward propagation in each layer.)

    Args:
        f (func): Function to compute gradient on.
        x (numpy.ndarray): An input to function for which we compute gradient on.
        df (numpy.ndarray): Upstream gradient.
        h (float): Small delta for computing gradient.
    Returns:
        grad (numpy.ndarray): Gradient of f with respect to x.
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