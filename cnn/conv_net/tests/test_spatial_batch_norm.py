# Created: March, 2018
# Author(s): Calvin Feng

from layer.spatial_batch_norm import SpatialBatchNorm
from layer.gradient_check import *
import numpy as np
import unittest


class SpatialBatchNormTest(unittest.TestCase):
    def test_forward_pass_train_mode(self):
        self.layer = SpatialBatchNorm()

        np.random.seed(1)
        N, C, H, W = 2, 3, 4, 5
        x = 4 * np.random.randn(N, C, H, W) + 10
        gamma = np.ones(C)
        beta = np.zeros(C)
        mode = 'train'

        norm_out = self.layer.forward_pass(x, gamma, beta)
        np.testing.assert_array_almost_equal(norm_out.mean(axis=(0, 2, 3)), np.zeros(C), decimal=6)
        np.testing.assert_array_almost_equal(norm_out.std(axis=(0, 2, 3)), np.ones(C), decimal=6)

    def test_forward_pass_test_mode(self):
        self.layer = SpatialBatchNorm()

        np.random.seed(1)
        N, C, H, W = 10, 4, 11, 12
        gamma = np.ones(C)
        beta = np.zeros(C)
        for t in range(50):
            x = 2.3 * np.random.randn(N, C, H, W) + 13
            self.layer.forward_pass(x, gamma, beta)

        x = 2.3 * np.random.randn(N, C, H, W) + 13
        norm_out = self.layer.forward_pass(x, gamma, beta, mode='test')

        # Sinc we are using running_mean and running_var, the precision should be much lower
        np.testing.assert_array_almost_equal(norm_out.mean(axis=(0, 2, 3)), np.zeros(C), decimal=1)
        np.testing.assert_array_almost_equal(norm_out.std(axis=(0, 2, 3)), np.ones(C), decimal=1)

    def test_backward_pass(self):
        self.layer = SpatialBatchNorm()

        np.random.seed(1)
        N, C, H, W = 2, 3, 4, 5
        x = 5 * np.random.randn(N, C, H, W) + 12
        gamma = np.random.randn(C)
        beta = np.random.randn(C)

        grad_out = np.random.randn(N, C, H, W)
        mode = 'train'

        self.layer.forward_pass(x, gamma, beta, mode)
        grad_x, grad_gamma, grad_beta = self.layer.backward_pass(grad_out)

        fx = lambda x: self.layer.forward_pass(x, gamma, beta, mode)
        fg = lambda g: self.layer.forward_pass(x, g, beta, mode)
        fb = lambda b: self.layer.forward_pass(x, gamma, b, mode)

        num_grad_x = eval_numerical_gradient_array(fx, x, grad_out)
        num_grad_gamma = eval_numerical_gradient_array(fg, gamma.copy(), grad_out)
        num_grad_beta = eval_numerical_gradient_array(fb, beta.copy(), grad_out)

        self.assertAlmostEqual(1e-8, rel_error(num_grad_x, grad_x), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_gamma, grad_gamma), places=2)
        self.assertAlmostEqual(1e-8, rel_error(num_grad_beta, grad_beta), places=2)
