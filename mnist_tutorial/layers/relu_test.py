from relu import ReLU
from gradient_check import eval_numerical_gradient_array

import numpy as np
import unittest


class ReLUTest(unittest.TestCase):
    def setUp(self):
        """Configures and sets up variables for each test case

        N (int): Number of inputs
        D (int): Input dimension
        """
        np.random.seed(314)

        self.N = 100
        self.D = 10
        self.layer = ReLU()

    def tearDown(self):
        """Tear down after each test case
        """
        pass

    def test_forward_prop(self):
        x = np.linspace(-1, 1, num=self.N*self.D).reshape(self.N, self.D)
        
        output = self.layer.forward_prop(x)
        expected_output = np.maximum(0, x)
        
        np.testing.assert_array_almost_equal(expected_output, output, decimal=7)

    def test_backprop(self):
        x = np.random.randn(self.N, self.D)
        grad_output = np.random.randn(*x.shape)
        
        # Numerical gradient w.r.t inputs
        num_grad_x = eval_numerical_gradient_array(f=lambda x: self.layer.forward_prop(x),
                                                   x=x, 
                                                   df=grad_output)

        # Compute gradients using backprop algorithm
        grad_x = self.layer.backprop(grad_output)

        np.testing.assert_array_almost_equal(num_grad_x, grad_x, decimal=7)
