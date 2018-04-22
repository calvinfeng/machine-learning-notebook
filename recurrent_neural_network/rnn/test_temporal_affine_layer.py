# Created: April, 2018
# Author(s): Calvin Feng

from temporal_affine_layer import TemporalAffineLayer
from gradient_check import eval_numerical_gradient_array, rel_error
import numpy as np
import unittest


class TemporalAffineLayerTest(unittest.TestCase):
    def test_backward(self):
        np.random.seed(271)

        N, T, H, M = 2, 3, 4, 5 

        # Create the layer
        layer = TemporalAffineLayer(H, M)
        W = np.random.randn(H, M)
        b = np.random.randn(M)

        # Create some arbitrary inputs
        x = np.random.randn(N, T, H)

        out = layer.forward(x, W=W, b=b)
        grad_out = np.random.randn(*out.shape)
        grad_x, grad_W, grad_b = layer.backward(grad_out)

        fx = lambda x: layer.forward(x, W=W, b=b)
        fw = lambda W: layer.forward(x, W=W, b=b)
        fb = lambda b: layer.forward(x, W=W, b=b)

        grad_x_num = eval_numerical_gradient_array(fx, x, grad_out)
        grad_W_num = eval_numerical_gradient_array(fw, W, grad_out)
        grad_b_num = eval_numerical_gradient_array(fb, b, grad_out)

        self.assertAlmostEqual(rel_error(grad_x_num, grad_x), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(grad_W_num, grad_W), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(grad_b_num, grad_b), 1e-9, places=2)
