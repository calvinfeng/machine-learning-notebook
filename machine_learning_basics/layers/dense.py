import numpy as np


class Dense:
    def __init__(self):
        self.x = None
        self.w = None
        self.b = None

    def __call__(self, x, w, b):
        """Perform forward propagation

        Args:
            x (np.ndarray): Input
            w (np.ndarray): Kernel weights
            b (np.ndarray): Biases

        Returns:
            np.ndarray: Output
        """
        self.x = x
        self.w = w
        self.b = b
        return x @ w + b

    def gradients(self, grad_out):
        """Perform back propagation and return gradients with respect to upstream loss function.

        Args:
            grad_out (np.ndarray): Gradient of loss with respect to output.

        Returns:
            np.ndarray: Gradient of loss with respect to x
            np.ndarray: Gradient of loss with respect to w
            np.ndarray: Gradient of loss with respect to b
        """
        if self.x is None:
            raise ValueError("layer must be forward propagated first")
        
        grad_x = grad_out @ self.w.T
        grad_w = self.x.T @ grad_out
        grad_b = np.sum(grad_out, axis=0)
        return grad_x, grad_w, grad_b
