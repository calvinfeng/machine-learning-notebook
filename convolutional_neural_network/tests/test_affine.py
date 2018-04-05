# Created: March, 2018
# Author(s): Calvin Feng

from layer.affine import Affine
from layer.gradient_check import *
import numpy as np
import unittest


class AffineTest(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 2
        self.input_shape = (4, 5, 6)
        self.output_dim = 3

        self.input_size = self.num_inputs * np.prod(self.input_shape)
        self.weight_size = self.output_dim * np.prod(self.input_shape)
        self.layer = Affine()

    def test_forward_pass(self):
        x = np.linspace(-0.1, 0.5, num=self.input_size).reshape(self.num_inputs, *self.input_shape)
        w = np.linspace(-0.2, 0.3, num=self.weight_size).reshape(np.prod(self.input_shape), self.output_dim)
        b = np.linspace(-0.3, 0.1, num=self.output_dim)

        output = self.layer.forward_pass(x, w, b)
        correct_output = np.array([[ 1.49834967,  1.70660132,  1.91485297], [ 3.25553199,  3.5141327,   3.77273342]])
        err = rel_error(output, correct_output)

        # np.testing.asssert_almost_equal(output, correct_output, decimal=7)
        # Error should be very close to 1e-9, or almost zero
        self.assertAlmostEqual(err, 1e-9, places=2)

    def test_bacward_pass(self):
        np.random.seed(231)
        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        num_dx = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b), x, dout)
        num_dw = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b), w, dout)
        num_db = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b), b, dout)

        dx, dw, db = self.layer.backward_pass(dout)
        self.assertAlmostEqual(rel_error(num_dx, dx), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_dw, dw), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_db, db), 1e-9, places=2)
