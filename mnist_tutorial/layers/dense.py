import numpy as np


class Dense(object):
    def __init__(self):
        self.input = None
        self.weight = None
        self.bias = None

    def forward_prop(self, x, w, b):
        """Performs forward propagation in the Dense layer.

        Args:
            x (np.ndarray): Input data in matrix form.
            w (np.ndarray): Weight matrix for the layer.
            b (np.ndarray): Bias array

        Returns:
            output (np.ndarray)
        """
        self.input = x
        self.weight = w
        self.bias = b

        D = np.prod(x.shape[1:])
        x_reshaped = x.reshape(x.shape[0], D)

        return np.dot(x_reshaped, w) + b

    def backprop(self, grad_output):
        """Performs back propagation in Dense layer.

        Args:
            grad_out: Upstream derivative
    
        Returns:
            grad_x: Gradients of upstream variable with respect to input matrix
            grad_w: Gradient of upstream variable with respect to weight matrix of shape (D, H)
            grad_b: Gradient of upstream variable with respect to bias vector of shape (H,)

        The shape changes depending on whether this is the initial layer. For example, if it is the
        first layer in the network, then grad_x has the shape (N, d_1, ..., d_k) and grad_w has
        (D, H). Otherwise, the grad_x would be (N x D) and grad_w would be (D x H).
        """
        if self.input is not None and self.weight is not None:
            D = np.prod(self.input.shape[1:])
            input_reshaped = self.input.reshape(self.input.shape[0], D)

            grad_w = np.dot(input_reshaped.T, grad_output)
            grad_x = np.dot(grad_output, self.weight.T).reshape(self.input.shape)
            grad_b = np.sum(grad_output.T, axis=1)

            return grad_x, grad_w, grad_b  


if __name__ == '__main__':
    dummy = Dense()
    
    # Input is N by D, where D is the input dimension.
    x = np.array([[1, 2], [1, 2]]) # Shape: (2, 2)

    # Weight is D by H, where H is the hidden unit dimension.  
    w = np.array([[1, 2, 3], [4, 5, 6]]) # Shape: (2, 3)

    # Bias is always 1 by D, where D is the input dimension.
    b = np.array([[1, 1, 1]]) # Shape: (1, 3)

    # What is the shape of output?
    print dummy.forward_prop(x, w, b)