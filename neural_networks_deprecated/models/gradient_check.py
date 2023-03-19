import numpy as np

#########################################################################################
# DEPRECATED - Logic of the gradient check has been moved into the neural network class #
#########################################################################################
def eval_numerical_gradient(f, x, h=1e-4):
    """Evaluate numerical gradient of f at x
    Args:
        f: Lambda function that takes a single argument
        x: Numpy array to evaluate the gradient at
    """
    fx = f(x) # Evaluate function value at original point
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Evaluate function at x + h
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + h
        fx_plus_h = f(x) # evaluate f(x + h)
        x[ix] = oldval - h
        fx_minus_h = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        grad[ix] = (fx_plus_h - fx_minus_h) / (2 * h) # Slope
        print (ix, grad[ix])
        it.iternext()

    return grad
