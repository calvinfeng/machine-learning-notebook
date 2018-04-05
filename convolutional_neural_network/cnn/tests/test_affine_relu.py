# Created: March, 2018
# Author(s): Calvin Feng

from composite.affine_relu import AffineReLU
from composite.gradient_check import *
import numpy as np
import unittest


class AffineReLUTest(unittest.TestCase):
    def setUp(self):
        self.layer = AffineReLU()

    def test_backward_pass(self):
        np.random.seed(231)
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        grad_out = np.random.randn(2, 10)

        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b), x, grad_out)
        num_grad_w = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b), w, grad_out)
        num_grad_b = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b), b, grad_out)

        grad_x, grad_w, grad_b = self.layer.backward_pass(grad_out)

        self.assertAlmostEqual(rel_error(num_grad_x, grad_x), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_w, grad_w), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_b, grad_b), 1e-9, places=2)
