# Created: March, 2018
# Author(s): Calvin Feng

from layer.max_pool import MaxPool
from layer.gradient_check import *
import numpy as np
import unittest


class MaxPoolTest(unittest.TestCase):
    def setUp(self):
        self.layer = MaxPool()

    def test_forward_pass(self):
        self.layer.pool_height = 2
        self.layer.pool_width = 2
        self.layer.stride = 2
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
        out = self.layer.forward_pass(x)
        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [ 0.03157895,  0.04631579]]],
                                [[[ 0.09052632,  0.10526316],
                                  [ 0.14947368,  0.16421053]],
                                 [[ 0.20842105,  0.22315789],
                                  [ 0.26736842,  0.28210526]],
                                 [[ 0.32631579,  0.34105263],
                                  [ 0.38526316,  0.4       ]]]])
        self.assertAlmostEqual(rel_error(out, correct_out), 1e-8, places=2)

    def test_backward_pass(self):
        self.layer.pool_height = 2
        self.layer.pool_width = 2
        self.layer.stride = 2
        np.random.seed(1)
        X = np.random.randn(3, 2, 8, 8)
        grad_out = np.random.randn(3, 2, 4, 4)

        func = lambda x: self.layer.forward_pass(x)
        num_grad_x = eval_numerical_gradient_array(func, X, grad_out)
        grad_x = self.layer.backward_pass(grad_out)

        self.assertAlmostEqual(rel_error(grad_x, num_grad_x), 1e-8, places=2)
