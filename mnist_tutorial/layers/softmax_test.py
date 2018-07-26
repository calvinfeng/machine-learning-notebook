from softmax import Softmax
from gradient_check import eval_numerical_gradient

import numpy as np
import unittest


def categorical_cross_entropy(y_pred, y):
    """Computes categorical cross entropy loss.

    Args:
        y_pred (numpy.ndarray): Output of the network, of shape (N, C) where x[i, j] is the softmax 
                                probability for for jth class for the ith input.
        y (numpy.ndarray): Vector of labels in one-hot representation.
    
    Returns:
        loss (float): Scalar value of the cross entropy loss.
    """
    N = len(y_pred)
    y = np.argmax(y, axis=1)
    log_probs = np.log(y_pred)

    return -1 * np.sum(log_probs[np.arange(N), y]) / N


class SoftmaxTest(unittest.TestCase):
    def setUp(self):
        """Configures and sets up variables for each test case

        N (int): Number of inputs
        D (int): Input dimension
        """
        np.random.seed(314)

        self.N = 10
        self.D = 10
        self.layer = Softmax()

    def tearDown(self):
        """Tear down after each test case
        """
        pass
    
    def test_forward_prop(self):
        x = np.linspace(-1, 1, num=self.N*self.D).reshape(self.N, self.D)
        output = self.layer.forward_prop(x)

        np.testing.assert_array_almost_equal(np.ones(self.N), np.sum(output, axis=1), decimal=7)
        np.testing.assert_array_almost_equal(np.ones(self.N) * (self.D - 1), np.argmax(output, axis=1), decimal=7)

    def test_backprop(self):
        x = np.random.randn(self.N, self.D)
        y = np.random.randn(*x.shape)

        # Numerical gradient w.r.t inputs
        num_grad_x = eval_numerical_gradient(f=lambda x: categorical_cross_entropy(self.layer.forward_prop(x), y),
                                             x=x,
                                             verbose=False)
        
        # Compute gradients using backprop algorithm
        grad_x = self.layer.backprop(y)

        np.testing.assert_array_almost_equal(num_grad_x, grad_x, decimal=7)


