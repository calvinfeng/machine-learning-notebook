import numpy as np


class Dense(object):
    def __init__(self):
        self.input = None
        self.weight = None
        self.bias = None

    def forward(self, x, w, b):
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
        
        return np.dot(x, w) + b


if __name__ == '__main__':
    dummy = Dense()
    
    # Input is N by D, where D is the input dimension.
    x = np.array([[1, 2], [1, 2]]) # Shape: (2, 2)

    # Weight is D by H, where H is the hidden unit dimension.  
    w = np.array([[1, 2, 3], [4, 5, 6]]) # Shape: (2, 3)

    # Bias is always 1 by D, where D is the input dimension.
    b = np.array([[1, 1, 1]]) # Shape: (1, 3)

    # What is the shape of output?
    print dummy.forward(x, w, b)