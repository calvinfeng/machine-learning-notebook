# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np
from layer.conv import Conv
from layer.relu import ReLU
from layer.max_pool import MaxPool


class ConvReLUMaxPool(object):
    def __init__(self, conv_param, pool_param):
        """
        Args:
            conv_param: A dictionary containing the following keys
                - stride: The number of pixels between adjacent receptive fields in the horizontal and vertical directions
                - pad: The number of pixels that will be used to zero-pad the input
            pool_param: A dictionary containing the following keys
                - pool_height: Height of each pooling region
                - pool_width: Width of each pooling region
                - stride: Distance between adjacent pooling regions
        """
        self.conv_layer = Conv(stride=conv_param['stride'], pad=conv_param['pad'])
        self.relu_layer = ReLU()
        self.max_pool_layer = MaxPool(pool_height=pool_param['pool_height'],
                                    pool_width=pool_param['pool_width'],
                                    stride=pool_param['stride'])

    def forward_pass(self, x, w, b):
        """
        Args:
            x: Input to convolutional layer
            w: Weights for convolutional layer
            b: Biases for convolutional layer

        Returns:
            max_pool_out: Output from the max pooling layer
        """
        conv_out = self.conv_layer.forward_pass(x, w, b)
        relu_out = self.relu_layer.forward_pass(conv_out)
        max_pool_out = self.max_pool_layer.forward_pass(relu_out)

        return max_pool_out

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out

        Returns:
            grad_x: Gradients w.r.t. input to convolutional layer
            grad_w: Gradient w.r.t. weights to convolutional layer
            grad_b: Gradient w.r.t. biases to convolutional layer
        """
        grad_max_pool = self.max_pool_layer.backward_pass(grad_out)
        grad_relu = self.relu_layer.backward_pass(grad_max_pool)
        grad_x, grad_w, grad_b = self.conv_layer.backward_pass(grad_relu)

        return grad_x, grad_w, grad_b
