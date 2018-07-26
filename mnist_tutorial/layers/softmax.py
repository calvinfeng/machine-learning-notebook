import numpy as np


class Softmax(object):
    def __init__(self):
        self.output = None
    
    def forward_prop(self, x):
        """Performs forward propagation in softmax activation layer.

        Args:
            x (numpy.ndarray): Input matrix of any shape
        
        Returns:
            output (numpy.ndarray): A matrix with softmax applied, same shape as input
        """
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        probs = np.exp(shifted_logits) / Z
        
        # Cache the output, i.e. softmax probabilities
        self.output = probs

        ### Math Trick ###
        # log(probs) can be computed with the following short-cut
        # log_probs = shifted_logits - np.log(Z)
        # Then we can just cache the output as 
        # self.output = np.exp(log_probs)

        return probs
    
    def backprop(self, y):
        """Performs backward propagation in softmax activation layer.

        Args:
            y (numpy.ndarray): Correct labels of each classes in -ne-hot encoding format.
        
        Returns:
            grad_in (numpy.ndarray): Gradient of cross entropy loss with respect to input scores.
        """
        N = len(y)
        grad_in = self.output.copy()
        grad_in[np.arange(N), np.argmax(y, axis=1)] -= 1
        grad_in /= N

        return grad_in