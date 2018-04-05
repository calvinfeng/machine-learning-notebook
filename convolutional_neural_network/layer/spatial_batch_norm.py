# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np
from batch_norm import BatchNorm


class SpatialBatchNorm(BatchNorm):
    """SpatialBatchNorm implements a network layer that performs spatial batch normalization.

    This is a slightly modified version of the original batch normalization. The expected input is of shape (N, C, H, W)
    which is N images. The normalization is performed across channels (C) or filters depending on where this layer is
    inserted in the network architecture. The first step is to flatten the N*H*W pixels into a matrix of shape (N*H*W, C).
    The mini-batch statistic is then performed on every pixel across C.
    """
    def __init__(self, **kwargs):
        """
        Keyword args:
            eps: constant for numeric stability
            momentum: constant for running mean/variance calculation
            running_mean: if input has shape (N, D), then this is array of shape (D,)
            running_var: if input has shape (N, D), then this is array of shape (D,)
        """
        super(SpatialBatchNorm, self).__init__(**kwargs)

    def forward_pass(self, x, gamma, beta, mode='train'):
        """
        Args:
            x: Input matrix of shape (N, C, H, W)
            gamma: Scale parameter of shape (C,)
            beta: Shift parameter of shape (C,)
            mode: 'train' or 'test'
        Returns:
            out: The output of batch normalization
        """
        # First, move height and width to the front and flatten the first three layers
        N, C, H, W = x.shape
        flatten_x = x.transpose(0, 2, 3, 1).reshape((N * H * W, C))

        # Perform usual batch norm and then restore the original shape
        flatten_out = super(SpatialBatchNorm, self).forward_pass(flatten_x, gamma, beta, mode=mode)
        out = flatten_out.reshape((N, H, W, C)).transpose(0, 3, 1, 2)

        return out

    def backward_pass(self, grad_out, simplified=True):
        """
        Args:
            grad_out: Upstream gradient
            simplified: If simplified, the gradient computation will use a simplified implementation
        Returns:
            grad_x: Gradient with respect to x, of shape (N, C, H, W)
            grad_gamma: Gradient with respect to gamma, of shape (C,)
            grad_beta: Gradient with respect to beta, of shape (C,)
        """
        # Same as above, move the height and width to the front and then flatten first three layers
        N, C, H, W = grad_out.shape
        flatten_grad_out = grad_out.transpose(0, 2, 3, 1).reshape((N * H * W, C))

        # Perform usual batch norm backprop and then restore the original shape
        flatten_grad_x, grad_gamma, grad_beta = super(SpatialBatchNorm, self).backward_pass(flatten_grad_out, simplified=simplified)
        grad_x = flatten_grad_x.reshape((N, H, W, C)).transpose(0, 3, 1, 2)

        return grad_x, grad_gamma, grad_beta
