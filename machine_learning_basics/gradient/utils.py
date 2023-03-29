import numpy as np


def numerical_gradient(f, x, grad_out, delta=1e-5):
    grad = np.zeros_like(x)
    iter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iter.finished:
        ij = iter.multi_index

        init_val = x[ij]

        x[ij] = init_val + delta
        pos = f(x).copy()

        x[ij] = init_val - delta
        neg = f(x).copy()

        x[ij] = init_val

        grad[ij] = np.sum((pos - neg) * grad_out) / (2 * delta)
        iter.iternext()
    return grad
