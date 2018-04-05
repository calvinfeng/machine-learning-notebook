# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np


class Affine(object):
    """Affine implements a network layer that performs matrix multiplication.

    Affine transformation is basically multiplying an input vector with a weight matrix plus bias. The input can come in
    any shape, (N, d1, d2, d3, ..., d_n), e.g. a set of 100 RGB 32x32 images is (100, 3, 32, 32).
    """
    def __init__(self):
        self.x = None
        self.w = None
        self.b = None

    def forward_pass(self, x, w, b):
        """
        Args:
            x: Input of any shape

        Returns:
            out: A matrix of product of the input and weights plus bias
        """
        self.x = x
        self.w = w
        self.b = b

        D = np.prod(x.shape[1:])
        x_tf = x.reshape(x.shape[0], D)
        out = np.dot(x_tf, w) + b

        return out

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out: Upstream derivative

        Returns:
            grad_x: Gradients of upstream variable with respect to input matrix
            grad_w: Gradient of upstream variable with respect to weight matrix of shape (D, M)
            grad_b: Gradient of upstream variable with respect to bias vector of shape (M,)

        The shape changes depending on which layer this gate is inserted. For example, if it is the first gate in the
        network, then grad_x has the shape (N, d_1, ..., d_k) and grad_w has (D, M). Otherwise, the grad_x would
        be (N x M) and grad_w would be (M x M).
        """
        if self.x is not None and self.w is not None:
            D = np.prod(self.x.shape[1:])
            input_tf = self.x.reshape(self.x.shape[0], D)

            grad_w = np.dot(input_tf.T, grad_out)
            grad_x = np.dot(grad_out, self.w.T).reshape(self.x.shape)
            grad_b = np.sum(grad_out.T, axis=1)

            return grad_x, grad_w, grad_b
