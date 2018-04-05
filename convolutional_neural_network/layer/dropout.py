# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np


class Dropout(object):
    """Dropout implements a network layer that performs drop out regularization
    """
    def __init__(self, **kwargs):
        """
        Keyword args:
            prob: Probability for each neuron to drop out
            seed: Seeding integer for random generator
        """
        self.mask = None
        self.mode = None

        # Define dropout parameters
        self.prob = kwargs.get('prob', 0)
        self.seed = kwargs.get('seed', None)

    def forward_pass(self, x, mode='train'):
        """
        Args:
            x: Input of any shape
            mode: 'test' or 'train', optional
        Returns:
            out: Output of the same shape as input
        """
        self.mode = mode
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.mode == 'train':
            self.mask = np.ones(x.shape)
            prob_arr = np.random.random(x.shape)
            self.mask[prob_arr <= self.prob] = 0
            out = self.mask * x
        elif self.mode == 'test':
            out = x
        else:
            raise ValueError("Invalid forward drop out mode: %s" % mode)

        out = out.astype(x.dtype, copy=False)
        return out

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out: Upstream gradient
        Returns:
            grad_x: Gradient w.r.t. input x
        """
        if self.mode == 'train':
            grad_out[self.mask == 0] = 0
            grad_x = grad_out
        elif self.mode == 'test':
            grad_x = grad_out
        else:
            raise ValueError("Invalid backward drop out mode: %s" % self.mode)

        return grad_x
