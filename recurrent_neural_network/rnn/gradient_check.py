import numpy as np


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """Evaluate a numeric gradient for a function that accepts a numpy array and returns a numpy array.
    
    Args:
        f (lambda): Lambda function that accepts x as an input
        x (np.array): Input of any shape, as long as f accepts it
        df (np.array): Upstream gradient
        h (float): Smaller number for taking limits

    Returns:
        grad (float): Gradient value    
    """

    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """Naive implementation of numerical gradient of f at x

    Args:
        f (lambda): Function that takes a single argument
        x (np.array): The point (numpy array) to evaluate the gradient at

    Returns:
        grad (float): Gradient value
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad