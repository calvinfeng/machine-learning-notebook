import numpy as np
import unittest
from neuralnet.network import NeuralNetwork

def generate_random_data(N, input_dim, output_dim):
    """ Generate a set random input vectors and output vectors.
    Args:
        N: Number of vectors
        input_dim: Dimension of the input vectors
        output_dim: Dimension of the output vectors
    """
    X = 10 * np.random.randn(N, input_dim)
    y = np.random.randint(output_dim, size=N)
    return X, y


# Run nosetests --nocapture
class ForwardPropTests(unittest.TestCase):
    def setUp(self):
        self.N, self.input_dim, self.hidden_dim, self.classes = 10, 4, 10, 5
        self.network = NeuralNetwork(self.input_dim, self.hidden_dim, self.classes, std=0.25)
        self.rand_X, self.rand_y = generate_random_data(self.N, self.input_dim, self.classes)

    def test_forward_prop(self):
        # Initializing the network, using a small standard deviation, the network vanishes very quickly!
        act = self.network._forward_prop(self.rand_X)
        print act['probs']
        for sum in np.sum(act['probs'], axis=1):
            self.assertAlmostEqual(sum, 1)

    def test_predict(self):
        pred_arr = self.network.predict(self.rand_X)
        print pred_arr
        for pred in pred_arr:
            self.assertTrue(0 <= pred <= self.N - 1)

    def test_loss(self):
        act = self.network._forward_prop(self.rand_X)
        loss = self.network._loss(self.rand_X, self.rand_y, act['probs'], 0)
        print loss
        self.assertTrue(isinstance(loss, float))
