import numpy as np


class Sigmoid:
    def __init__(self):
        self.x = None

    def __call__(self, x):
        """Perform forward propagation

        Args:
            x (np.ndarray): Input

        Returns:
            np.ndarray: Output
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def gradients(self, grad_out):
        """Perofrm back propagation and return gradients with respect to upstream loss function.

        Args:
            grad_out (np.ndarray): Gradient of loss with respect to output.

        Returns:
            np.ndarray: Gradient of loss with respect to x
        """
        if self.x is None:
            raise ValueError("layer must be forward propagated first")
        
        y = self(self.x)
        return (1 - y) * y * grad_out
