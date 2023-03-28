import numpy as np


class ReLU:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        out = x.copy()
        out[x < 0] = 0
        return out

    def gradients(self, grad_out):
        if self.x is None:
            raise ValueError("layer must be forward propagated first")

        grad_x = grad_out.copy()
        grad_x[self.x < 0] = 0
        return grad_x


class Sigmoid:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return 1 / (1 - np.exp(-x))

    def gradients(self, grad_out):
        if self.x is None:
            raise ValueError("layer must be forward propagated first")
        
        y = self(self.x)
        return (1 - y) * y * grad_out
