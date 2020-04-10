# Created: April, 2018
# Author(s): Calvin Feng

import numpy as np
import unittest
from temporal_softmax import temporal_softmax_loss
from gradient_check import eval_numerical_gradient, rel_error


class TemporalSoftmaxTest(unittest.TestCase):
    def test_temporal_softmax_loss(self):
        N, T, V = 7, 8, 9
        score = np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = (np.random.rand(N, T) > 0.5)
        
        _, grad_score = temporal_softmax_loss(score, y, mask, verbose=False)
        grad_score_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], score, verbose=False)

        self.assertAlmostEqual(rel_error(grad_score, grad_score_num), 1e-9, places=2)