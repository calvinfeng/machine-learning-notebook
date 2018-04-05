# Created: March, 2018
# Author(s): Calvin Feng

from layer.conv import Conv
from layer.gradient_check import *
import numpy as np
import unittest


class ConvTest(unittest.TestCase):
    def test_forward_pass(self):
        # Instantiate the layer
        self.layer = Conv(stride=2, pad=1)
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)
        out = self.layer.forward_pass(x, w, b)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]])
        self.assertAlmostEqual(1e-8, rel_error(out, correct_out), places=2)

    def test_backward_pass(self):
        # Instantiate the layer
        self.layer = Conv(stride=1, pad=1)

        np.random.seed(1)
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2,)
        grad_out = np.random.randn(4, 2, 5, 5)

        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b), x, grad_out)
        num_grad_w = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b), w, grad_out)
        num_grad_b = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b), b, grad_out)

        grad_x, grad_w, grad_b = self.layer.backward_pass(grad_out)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_x, grad_x), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_w, grad_w), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_b, grad_b), places=2)
