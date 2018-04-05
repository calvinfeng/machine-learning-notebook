# Created: March, 2018
# Author(s): Calvin Feng

from composite.affine_batch_norm_relu import AffineBatchNormReLU
from composite.gradient_check import *
import numpy as np
import unittest


class AffineBatchNormReLU(unittest.TestCase):
    def setUp(self):
        self.layer = AffineBatchNormReLU()

    def test_backward_pass(self):
        np.random.seed(1)
        N = 10
        output_dim = 10
        x = np.random.randn(N, 3, 4)

        D = x.shape[1] * x.shape[2]
        w = np.random.randn(D, output_dim)
        b = np.random.randn(output_dim)

        gamma = np.random.randn(output_dim)
        beta = np.random.randn(output_dim)

        mode = 'train'
        grad_out = np.random.randn(N, output_dim)
        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b, gamma, beta, mode), x, grad_out)
        num_grad_w = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b, gamma, beta, mode), w, grad_out)
        num_grad_b = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b, gamma, beta, mode), b, grad_out)
        num_grad_gamma = eval_numerical_gradient_array(lambda gamma: self.layer.forward_pass(x, w, b, gamma, beta, mode), gamma, grad_out)
        num_grad_beta = eval_numerical_gradient_array(lambda beta: self.layer.forward_pass(x, w, b, gamma, beta, mode), beta, grad_out)

        grad_x, grad_w, grad_b, grad_gamma, grad_beta = self.layer.backward_pass(grad_out)

        self.assertAlmostEqual(rel_error(num_grad_x, grad_x), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_w, grad_w), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_b, grad_b), 1e-9, places=2)

        # Don't forget to sum up the array obtained from numerical approximation
        self.assertAlmostEqual(rel_error(num_grad_gamma.sum(), grad_gamma), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_grad_beta.sum(), grad_beta), 1e-9, places=2)
