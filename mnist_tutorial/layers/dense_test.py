from dense import Dense
import numpy as np
import unittest


class DenseTest(unittest.TestCase):
    def setUp(self):
        self.N = 2 # Number of inputs
        self.D = 3 # Input dimension
        self.H = 4 # Output/Hidden dimension

        self.layer = Dense() # Define the layer that we are testing.
        
    def tearDown(self):
        pass

    def test_one_plus_one(self):
        self.assertEqual(1 + 1, 2)

    def test_one_plus_two(self):
        self.assertEqual(1 + 2, 3)

    def test_forward(self):
        x = np.linspace(-1, 1, num=self.N * self.D).reshape(self.N, self.D)
        w = np.linspace(-0.5, 0.5, num=self.D * self.H).reshape(self.D, self.H)
        b = np.linspace(-0.5, 0.5, num=self.H)

        output = self.layer.forward(x, w, b)
        expected_output = np.dot(x, w) + b
        
        np.testing.assert_array_almost_equal(expected_output, output, decimal=9)



