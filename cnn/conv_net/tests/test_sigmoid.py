# Created: March, 2018
# Author(s): Calvin Feng

from layer.sigmoid import Sigmoid
from layer.gradient_check import *
import numpy as np
import unittest


class SigmoidTest(unittest.TestCase):
    def setUp(self):
        self.layer = Sigmoid()

    def test_forward_pass(self):
        x = np.linspace(-1, 1, num=12).reshape(3, 4)
        output = self.layer.forward_pass(x)
        expected_output = np.array([[ 0.26894142,  0.30614975,  0.34606901,  0.38828059],
                                    [ 0.43223768,  0.47728837,  0.52271163,  0.56776232],
                                    [ 0.61171941,  0.65393099,  0.69385025,  0.73105858]])

        self.assertAlmostEqual(rel_error(output, expected_output), 1e-9, places=2)

    def test_backward_pass(self):
        np.random.seed(231)
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x), x, dout)
        grad_x = self.layer.backward_pass(dout)

        print grad_x
        self.assertAlmostEqual(rel_error(num_grad_x, grad_x), 1e-9, places=2)
