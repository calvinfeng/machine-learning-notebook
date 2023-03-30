import numpy as np


class Conv2D:
    def __init__(self, stride=1, padding=0):
        self.x = None
        self.padded_x = None
        self.w = None
        self.b = None
        self.stride = stride
        self.padding = padding

    def _output_dimension(self, input_dim, filter_dim):
        return int(1 + (input_dim + 2 * self.padding - filter_dim) / self.stride)

    def __call__(self, x, w, b):
        if len(x.shape) != 4:
            raise ValueError("input must be of shape (N, C, H, W)")

        if len(w.shape) != 4:
            raise ValueError("kernel must be of shape (F, C, H, W)")

        if w.shape[1] != x.shape[1]:
            raise ValueError("channel size must match")

        N, _, H, W = x.shape
        F, _, FH, FW = w.shape

        p, stride = self.padding, self.stride
        pad_dims = [(0, 0), (0, 0), (p, p), (p, p)]
        padded_x = np.pad(x, pad_width=pad_dims, mode='constant', constant_values=0)

        H_out = self._output_dimension(H, FH)
        W_out = self._output_dimension(W, FW)

        out = np.zeros((N, F, H_out, W_out))
        for n in range(N):
            for f in range(F):
                for i_out in range(H_out):
                    i_start = i_out * stride # Jump forward by stride size
                    i_end = i_start + FH
                    for j_out in range(W_out):
                        j_start = j_out * stride # Same here
                        j_end = j_start + FW
                        # Perform a convolution operation, sum across all channels
                        conv_sum = np.sum(padded_x[n, :, i_start:i_end, j_start:j_end] * w[f])
                        out[n, f, i_out, j_out] = conv_sum + b[f]

        self.x, self.padded_x = x, padded_x
        self.w, self.b = w, b
        return out

    def gradients(self, grad_out):
        """
        The back propagation for a convolution operation is also a convolution, but with
        spatially flipped filters.

        Suppose 
            - input has shape (1, 3, 2, 2) 
            - kernel has two 1x1 filters with 3-channels, (2, 3, 1, 1).
            - output has shape (1, 2, 2, 2) because of 2 filters.

        We have 6 weight values, w[0]: (3, 1, 1) and w[1]: (3, 1, 1). We slide each across the input (2, 2)
        grad_w[0][C][0][0] = x[0][C][0][0] + x[0][C][0][1] + x[0][C][1][0] + x[0][C][1][1]
        - Repeat for the other 2 channels
        
        Since the second filter also operates on the same input, the calculation is the same.
        """
        if self.x is None:
            raise ValueError("layer must be forward propagated first")

        grad_w = np.zeros(self.w.shape)
        grad_x = np.zeros(self.x.shape)
        grad_b = np.zeros(self.b.shape)

        N, _, H, W = self.x.shape
        F, _, FH, FW = self.w.shape
        H_out = self._output_dimension(H, FH)
        W_out = self._output_dimension(W, FW)
        stride = self.stride

        grad_padded_x = np.zeros(self.padded_x.shape)
        for n in range(N):
            for f in range(F):
                for i_out in range(H_out):
                    i_start = i_out * stride
                    i_end = i_start + FH
                    for j_out in range(W_out):
                        j_start = j_out * stride
                        j_end = j_start + FW
                        # No sum, we keep the kernel shape as it is.
                        grad_w[f, :, :, :] += self.padded_x[n, :, i_start:i_end, j_start:j_end] * grad_out[n, f, i_out, j_out]
                        grad_padded_x[n, :, i_start:i_end, j_start:j_end] += self.w[f, :, :, :] * grad_out[n, f, i_out, j_out]

        # Get rid of paddings
        grad_x = grad_padded_x[:, :, self.padding:self.padding+H, self.padding:self.padding+W]

        # Compute bias for each filter
        for f in range(F):
            grad_b[f] = np.sum(grad_out[:, f, :, :])

        return grad_x, grad_w, grad_b
