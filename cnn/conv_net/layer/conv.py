# Created: March, 2018
# Author(s): Calvin Feng

import numpy as np


class Conv(object):
    """Conv implements a network layer that performs convolution operation on input data

    Convolution expects an input consisting of N data points, each with C channels, height H and width W. It applies
    F different filters, where each filter spans all C channels and has height Hf and width Wf.
    """
    def __init__(self, **kwargs):
        """
        Keyword args:
            stride: The number of pixels between adjacent receptive fields in the horizontal and vertical directions
            pad: The number of pixels that will be used to zero-pad the input
        """
        self.x = None
        self.x_pad = None
        self.w = None
        self.b = None
        self.stride = kwargs.get('stride', 1)
        self.pad = kwargs.get('pad', 0)

    def forward_pass(self, x, w, b):
        """Naive implementation of forward pass for a convolutional layer, i.e. it has poor performance as compared to
        the native C implementation

        Args:
            x: Input data, of shape (N, C, H, W)
            w: Filter weights, of shape (F, C, Hf, Wf)
            b: Biases, of shape (F,)

        Returns:
            out: Output data, of shape (N, F, H_out, W_out)
        """
        pad, stride = self.pad, self.stride
        N, _, H, W = x.shape
        F, _, Hf, Wf = w.shape

        # API for pad_width is ((before_1, after_1), ..., (before_N, after_N)), we are only padding the image height and
        # width, that means we don't need to worry about the first 2 dimensions.
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad))
        self.x_pad = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

        H_out = int(1 + (H + 2 * pad - Hf) / stride)
        W_out = int(1 + (W + 2 * pad - Wf) / stride)
        out = np.zeros((N, F, H_out, W_out))

        # Now we iterate through every coordinate of the output and perform convolution on blocks of the padded input
        for n in range(N):
            for f in range(F):
                for h_out in range(H_out):
                    h_in = h_out * stride
                    for w_out in range(W_out):
                        w_in = w_out * stride
                        conv_sum = np.sum(self.x_pad[n][:, h_in:h_in + Hf, w_in:w_in + Wf] * w[f])
                        out[n, f, h_out, w_out] += conv_sum + b[f]

        # Caching important variables
        self.x, self.w, self.b = x, w, b
        return out

    def backward_pass(self, grad_out):
        """Naive implementation of backward pass for a convolutional layer, i.e. it has poor performanced as compared to
        the native C implementation.

        Args:
            grad_out: Upstream gradients

        Returns:
            grad_x: Gradient w.r.t. input
            grad_w: Gradient w.r.t. weights
            grad_b: Gradient w.r.t. biases
        """
        # The backward pass for a convolution operation (for both the data and the weights) is also a convolution, but
        # with spatially-flipped filters. It is easy to derive using 1 dimensional example.
        #
        # Suppose your x, of shape (3, 2, 2) and filter, of shape, (3, 1, 1) with stride = 1 and no padding. We have
        # three numbers that constitute the filter weight matrix, w[0], w[1] and w[3]. We can easily see that,
        # grad_w[0] = x[0][0][0] + x[0][0][1] + x[0][1][0] + x[0][1][1]
        # grad_w[1] = x[1][0][0] + x[1][0][1] + x[1][1][0] + x[1][1][1]
        # grad_w[2] = x[2][0][0] + x[2][0][1] + x[2][1][0] + x[2][1][1]
        #
        # Essentially, the filter slides across top-left, top-right, bottom-left, and bottom-right pixels. We sum up the
        # derivative contribution from each stride we make for each channel of the filter.
        if self.x is not None and self.x_pad is not None:
            grad_w = np.zeros(self.w.shape)
            grad_x = np.zeros(self.x.shape)
            grad_b = np.zeros(self.b.shape)

            N, _, H, W = self.x.shape
            F, _, Hf, Wf = self.w.shape
            _, _, H_out, W_out = grad_out.shape

            grad_x_pad = np.zeros(self.x_pad.shape)
            for n in range(N):
                for f in range(F):
                    for h_out in range(H_out):
                        h_in = h_out * self.stride
                        for w_out in range(W_out):
                            w_in = w_out * self.stride
                            # Summing up contribution from each filter block for each channel
                            grad_w[f] += self.x_pad[n][:, h_in:h_in + Hf, w_in:w_in + Wf] * grad_out[n, f, h_out, w_out]
                            grad_x_pad[n][:, h_in:h_in + Hf, w_in:w_in + Wf] += self.w[f] * grad_out[n, f, h_out, w_out]

            # Get rid of padding
            grad_x = grad_x_pad[:, :, self.pad:self.pad + H, self.pad:self.pad + W]

            for f in range(F):
                grad_b[f] = np.sum(grad_out[:, f, :, :])

            return grad_x, grad_w, grad_b
