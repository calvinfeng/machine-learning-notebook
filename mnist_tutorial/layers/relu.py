import numpy as np


class ReLU(object):
    def __init__(self):
        self.input = None

    def forward_prop(self, x):
        """Performs forward propagation in ReLU activation layer.

        Args:
            x (numpy.ndarray): Input matrix of any shape
            
        Returns:
            output: A matrix with ReLU applied, same shape as input
        """
        self.input = x
        
        return np.maximum(0, self.input)

    def backprop(self, grad_out):
        """Performs back propagation in ReLU activation layer.

        Args:
            grad_out (numpy.ndarray): Upstream derivative
        
        Returns:
            grad_in (numpy.ndarray): Gradient of upstream variable with respect to input
        """
        grad_in = grad_out
        grad_in[self.input<=0] = 0

        return grad_in