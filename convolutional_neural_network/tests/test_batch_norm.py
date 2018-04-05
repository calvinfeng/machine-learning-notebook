# Created: March, 2018
# Author(s): Calvin Feng

from layer.batch_norm import BatchNorm
from layer.gradient_check import *
import numpy as np
import unittest


class BatchNormTest(unittest.TestCase):
    def test_forward_pass(self):
        # Reset the running_mean and running_var by re-instantiation
        self.layer = BatchNorm()

        np.random.seed(1)
        N, D1, D2, D3 = 200, 50, 60, 3
        W1 = np.random.randn(D1, D2)
        W2 = np.random.randn(D2, D3)

        gamma = np.ones(D3)
        beta = np.zeros(D3)
        mode = 'train'

        for t in range(50):
            x = np.random.randn(N, D1)
            acts = np.maximum(0, x.dot(W1)).dot(W2)
            self.layer.forward_pass(acts, gamma, beta, mode)

        x = np.random.randn(N, D1)
        acts = np.maximum(0, x.dot(W1)).dot(W2)
        norm_x = self.layer.forward_pass(acts, gamma, beta, mode)

        np.testing.assert_array_almost_equal(norm_x.mean(axis=0), np.zeros(D3), decimal=6)
        np.testing.assert_array_almost_equal(norm_x.std(axis=0), np.ones(D3), decimal=6)

    def test_backward_pass(self):
        # Reset the running_mean and running_var by re-instantiation
        self.layer = BatchNorm()

        np.random.seed(1)
        N, D = 4, 5
        x = 5 * np.random.randn(N, D) + 12
        beta = np.random.randn(D)
        gamma = np.random.randn(D)
        grad_out = np.random.randn(N, D)

        mode = 'train'
        fx = lambda x: self.layer.forward_pass(x, gamma, beta, mode)
        fg = lambda g: self.layer.forward_pass(x, g, beta, mode)
        fb = lambda b: self.layer.forward_pass(x, gamma, b, mode)

        num_grad_x = eval_numerical_gradient_array(fx, x, grad_out)
        num_grad_gamma = eval_numerical_gradient_array(fg, gamma.copy(), grad_out)
        num_grad_beta = eval_numerical_gradient_array(fb, beta.copy(), grad_out)

        self.layer.forward_pass(x, gamma, beta, mode)

        # Test simplified implementation of gradient calculation
        grad_x, grad_gamma, grad_beta = self.layer.backward_pass(grad_out)

        self.assertAlmostEqual(1e-8, rel_error(num_grad_x, grad_x), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_gamma, grad_gamma), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_beta, grad_beta), places=2)

        # Test the chain rule implementation of gradient calculation
        grad_x, grad_gamma, grad_beta = self.layer.backward_pass(grad_out, simplified=False)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_x, grad_x), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_gamma, grad_gamma), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_beta, grad_beta), places=2)
