# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np
from layer.affine import Affine
from layer.relu import ReLU


class AffineReLU(object):
    def __init__(self):
        self.affine_layer = Affine()
        self.relu_layer = ReLU()

    def forward_pass(self, x, w, b):
        """Performs forward propagation through affine and ReLU layers

        Args:
            x: Input
            w: Weights
            b: Bias

        Returns:
            relu_out: Output from ReLU layer
        """
        affine_out = self.affine_layer.forward_pass(x, w, b)
        relu_out = self.relu_layer.forward_pass(affine_out)

        return relu_out

    def backward_pass(self, grad_out):
        """Performs back propagation through affine and ReLU layers

        Args:
            grad_out: Upstream gradient

        Returns:
            grad_x: Gradient w.r.t. input
            grad_w: Gradient w.r.t. weight
            grad_b: Gradient w.r.t. bias
        """
        grad_relu = self.relu_layer.backward_pass(grad_out)
        grad_x, grad_w, grad_b = self.affine_layer.backward_pass(grad_relu)

        return grad_x, grad_w, grad_b
