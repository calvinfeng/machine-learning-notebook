# Created: March, 2018
# Author(s): Calvin Feng

from composite.affine_relu_dropout import AffineReLUDropout
from composite.gradient_check import *
import numpy as np
import unittest


class AffineReLUDropoutTest(unittest.TestCase):
    def test_backward_pass(self):
        self.layer = AffineReLUDropout(dropout_param={ 'prob': 0.8, 'seed': 1 })

        np.random.seed(231)
        x = np.random.randn(100, 6, 6)
        w = np.random.randn(36, 10)
        b = np.random.randn(10)
        grad_out = np.random.randn(100, 10)

        out = self.layer.forward_pass(x, w, b, mode='train')
        grad_x, grad_w, grad_b = self.layer.backward_pass(grad_out)
        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b, mode='train'), x, grad_out)
        num_grad_w = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b, mode='train'), w, grad_out)
        num_grad_b = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b, mode='train'), b, grad_out)

        self.assertAlmostEqual(rel_error(num_grad_x, grad_x), 1e-6, places=2)
        self.assertAlmostEqual(rel_error(num_grad_w, grad_w), 1e-6, places=2)
        self.assertAlmostEqual(rel_error(num_grad_b, grad_b), 1e-6, places=2)
