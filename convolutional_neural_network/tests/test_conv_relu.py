# Created: March, 2018
# Author(s): Calvin Feng

from composite.conv_relu import ConvReLU
from composite.gradient_check import *
import numpy as np
import unittest


class ConvReLUTest(unittest.TestCase):
    def test_backward_pass(self):
        conv_param = {
        'stride': 1,
        'pad': 1
        }

        self.layer = ConvReLU(conv_param)

        np.random.seed(1)
        x = np.random.randn(2, 3, 8, 8)
        w = np.random.randn(3, 3, 3, 3)
        b = np.random.randn(3,)

        out = self.layer.forward_pass(x, w, b)
        grad_out = np.random.randn(2, 3, 8, 8)
        grad_x, grad_w, grad_b = self.layer.backward_pass(grad_out)

        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b), x, grad_out)
        num_grad_w = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b), w, grad_out)
        num_grad_b = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b), b, grad_out)

        self.assertAlmostEqual(rel_error(num_grad_x, grad_x), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_w, grad_w), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_b, grad_b), 1e-9, places=2)
