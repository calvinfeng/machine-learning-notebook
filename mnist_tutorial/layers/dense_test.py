from dense import Dense
from gradient_check import eval_numerical_gradient_array

import numpy as np
import unittest


class DenseTest(unittest.TestCase):
    def setUp(self):
        """Configure and setup variables for each test case

        N (int): Number of inputs
        D (int): Input dimension, which is the total number of pixels from each input image.
        H (int): Output/hidden unit dimension
        """
        np.random.seed(19880911)

        self.img_height = 28
        self.img_width = 28

        self.N = 10
        self.D = self.img_height * self.img_width 
        self.H = 10
        self.layer = Dense()
        
    def tearDown(self):
        """Tear down after each test case
        """
        pass

    def test_forward_prop(self):
        x = np.linspace(-1, 1, num=self.N * self.D).reshape(self.N, self.img_height, self.img_width)
        w = np.linspace(-0.5, 0.5, num=self.D * self.H).reshape(self.D, self.H)
        b = np.linspace(-0.5, 0.5, num=self.H)

        output = self.layer.forward_prop(x, w, b)
        expected_output = np.dot(x.reshape(self.N, self.D), w) + b
        
        np.testing.assert_array_almost_equal(expected_output, output, decimal=7)

    def test_backprop(self):
        x = np.random.randn(self.N, self.img_height, self.img_width)
        w = np.random.randn(self.D, self.H)
        b = np.random.randn(self.H)
        grad_output = np.random.randn(self.N, self.H)

        # Numerical gradient w.r.t inputs
        num_grad_x = eval_numerical_gradient_array(f=lambda x: self.layer.forward_prop(x, w, b), 
                                                   x=x, 
                                                   df=grad_output)
        # Numerical gradient w.r.t weights
        num_grad_w = eval_numerical_gradient_array(f=lambda w: self.layer.forward_prop(x, w, b),
                                                   x=w, 
                                                   df=grad_output)

        # Numerical gradient w.r.t. biases
        num_grad_b = eval_numerical_gradient_array(f=lambda b: self.layer.forward_prop(x, w, b),
                                                   x=b, 
                                                   df=grad_output)

        # Compute gradient using backprop algorithm.
        grad_x, grad_w, grad_b = self.layer.backprop(grad_output)

        np.testing.assert_array_almost_equal(num_grad_x, grad_x, decimal=7)
        np.testing.assert_array_almost_equal(num_grad_w, grad_w, decimal=7)
        np.testing.assert_array_almost_equal(num_grad_b, grad_b, decimal=7)


