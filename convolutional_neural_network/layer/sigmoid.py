# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np


class Sigmoid(object):
    """Sigmoid implements a network layer that performs sigmoid activation which squashes input to a range (0, 1)
    """
    def __init__(self):
        self.output = None

    def forward_pass(self, input):
        """
        Args:
            input: A matrix of any shape

        Returns:
            output: A matrix with sigmoid activation applied, same shape as input
        """
        self.output = 1.0 / (1.0 + np.exp(-1.0 * input))
        return self.output.copy()

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out: Upstream derivative

        Returns:
            grad_in: Gradient of upstream variable with respect to input
        """

        grad_in = self.output * ( 1.0 - self.output) * grad_out
        return grad_in
