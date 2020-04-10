# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np
from layer.affine import Affine
from layer.relu import ReLU
from layer.batch_norm import BatchNorm


class AffineBatchNormReLU(object):
    def __init__(self, batch_norm_param=None):
        """
        Optional argument:
            batch_norm_param: A dictionary containing the following keys
                - eps: constant for numeric stability, required
                - momentum: constant for running mean/variance calculation, required
                - running_mean: if input has shape (N, D), then this is array of shape (D,)
                - running_var: if input has shape (N, D), then this is array of shape (D,)
        """
        self.affine_layer = Affine()
        self.relu_layer = ReLU()
        if batch_norm_param is not None:
            self.batch_norm_layer = BatchNorm(eps=batch_norm_param['eps'],
                                            momentum=batch_norm_param['momentum'],
                                            running_mean=batch_norm_param.get('running_mean', None),
                                            running_var=batch_norm_param.get('running_var', None))
        else:
            self.batch_norm_layer = BatchNorm()

    def forward_pass(self, x, w, b, gamma, beta, mode='train'):
        """ Performs forward propagation through affine, batch normalization, and rectinfied linear unit layers

        Args:
            x: Input
            w: Weights
            b: Bias
            gamma: Scale factor
            beta: Shifting factor
            mode: 'train' or 'test'

        Returns:
            relu_out: Output from ReLU layer
        """
        affine_out = self.affine_layer.forward_pass(x, w, b)
        batch_norm_out = self.batch_norm_layer.forward_pass(affine_out, gamma, beta, mode)
        relu_out = self.relu_layer.forward_pass(batch_norm_out)

        return relu_out

    def backward_pass(self, grad_out):
        """Performs back propagation through affine, batch normalization, and rectinfied linear unit layers

        Args:
            grad_out: Upstream gradient

        Returns:
            grad_x: Gradient w.r.t. input
            grad_w: Gradient w.r.t. weight
            grad_b: Gradient w.r.t. bias
            grad_gamma: Gradient w.r.t. gamma constant
            grad_beta: Gradient w.r.t. beta constant
        """
        grad_relu = self.relu_layer.backward_pass(grad_out)
        grad_batch_norm, grad_gamma, grad_beta = self.batch_norm_layer.backward_pass(grad_relu)
        grad_x, grad_w, grad_b = self.affine_layer.backward_pass(grad_batch_norm)

        return grad_x, grad_w, grad_b, np.sum(grad_gamma), np.sum(grad_beta)
