# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np
from layer.affine import Affine
from layer.relu import ReLU
from layer.dropout import Dropout


class AffineReLUDropout(object):
    def __init__(self, dropout_param=None):
        """
        Args:
            dropout_param: A dictionary with the following key(s):
                - prob: Probability for each neuron to drop out, required
                - seed: Seeding integer for random generator, optional
        """
        self.affine_layer = Affine()
        self.relu_layer = ReLU()
        if dropout_param is not None:
            self.dropout_layer = Dropout(prob=dropout_param['prob'], seed=dropout_param.get('seed', None))
        else:
            self.dropout_layer = Dropout()

    def forward_pass(self, x, w, b, mode='train'):
        """ Performs forward propagation through affine, rectinfied linear unit, and dropout layers

        Args:
            x: Input
            w: Weights
            b: Bias
            mode: 'train' or 'test'

        Returns:
            dropout_out: Output from Dropout layer
        """
        affine_out = self.affine_layer.forward_pass(x, w, b)
        relu_out = self.relu_layer.forward_pass(affine_out)
        dropout_out = self.dropout_layer.forward_pass(relu_out, mode)

        return dropout_out

    def backward_pass(self, grad_out):
        """Performs back propagation through  affine, rectinfied linear unit, and dropout layers

        Args:
            grad_out: Upstream gradient

        Returns:
            grad_x: Gradient w.r.t. input
            grad_w: Gradient w.r.t. weight
            grad_b: Gradient w.r.t. bias
        """
        grad_dropout = self.dropout_layer.backward_pass(grad_out)
        grad_relu = self.relu_layer.backward_pass(grad_dropout)
        grad_x, grad_w, grad_b = self.affine_layer.backward_pass(grad_relu)

        return grad_x, grad_w, grad_b
