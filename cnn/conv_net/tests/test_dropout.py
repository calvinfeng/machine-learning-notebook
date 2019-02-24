# Created: March, 2018
# Author(s): Calvin Feng

from layer.dropout import Dropout
from layer.gradient_check import *
import numpy as np
import unittest


class DropoutTest(unittest.TestCase):
    def setUp(self):
        self.layer = Dropout()

    def test_forward_pass(self):
        np.random.seed(1)
        x = np.random.randn(500, 500) + 10

        for p in [0.3, 0.6, 0.75]:
            self.layer.prob = p
            out_train = self.layer.forward_pass(x, mode='train')
            out_test = self.layer.forward_pass(x, mode='test')

            out_train_mean = out_train.mean()
            out_test_mean = out_test.mean()
            self.assertTrue(out_train_mean < out_test_mean)

            zero_fraction_train = (out_train == 0).mean()
            zero_fraction_test = (out_test == 0).mean()
            self.assertAlmostEqual(zero_fraction_train, p, places=2)
            self.assertAlmostEqual(zero_fraction_test, 0, places=2)

    def test_backward_pass(self):
        self.layer.prob = 0.8
        self.layer.seed = 1
        np.random.seed(1)
        X = np.random.randn(10, 10) + 10
        grad_out = np.random.randn(*X.shape)
        num_grad_x = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, mode='train'), X, grad_out)
        grad_x = self.layer.backward_pass(grad_out)
        self.assertAlmostEqual(rel_error(num_grad_x, grad_x), 1e-8, places=-2)
