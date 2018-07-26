import numpy as np


class ReLU(object):
    """ReLU implements a network layer that performs rectified linear unit transformation.
    The input to this layer is typically, but not limited to, a matrix of shape (N, D) . The layer 
    squashes all values that are less than zero in input matrix.
    """
    def __init__(self):
        self.input = None

    def forward_prop(self, x):
        """Performs forward propagation in ReLU activation layer.

        Args:
            input: A matrix of any shape
            
        Returns:
            output: A matrix with ReLU applied, same shape as input
        """
        self.input = x
        
        return np.maximum(0, self.input)

    def backprop(self, grad_out):
        """Performs back propagation in ReLU activation layer.

        Args:
            grad_out: Upstream derivative
        
        Returns:
            grad_in: Gradient of upstream variable with respect to input
        """
        grad_in = grad_out
        grad_in[self.input<=0] = 0

        return grad_in