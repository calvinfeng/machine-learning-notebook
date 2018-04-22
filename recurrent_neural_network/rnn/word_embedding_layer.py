# Created: April, 2018
# Author(s): Calvin Feng

import numpy as np


class WordEmbeddingLayer(object):
    """Word embedding layer enables us to present words using vectors. Each word of the vocabulary 
    be associated with a vector and these vectors will be learned jointly with the rest of the system.
    """
    def __init__(self, vocab_size, word_vec_dim):
        """
        Args: 
            vocab_size (int): Number of words in the dictionary.
            word_vec_dim (int): Dimension of the desired word vector.
        """
        self.V = vocab_size
        self.D = word_vec_dim
        self.W = np.random.randn(self.V, self.D)
        self.W /= 100
        self.x = None

    def forward(self, x, W=None):
        """Forward pass for word embedding layer. This function operates on mini-batch of size N 
        where each sequence has length T.

        Args:
            x (np.array): Integer array of shape (N, T), each element idx of x must be in the range 0 <= idx < V
        
        Returns:
            np.array: Array of shape (N, T, D) which represents the word vectors
        """
        if W is not None:
            self.W = W

        self.x = x
        return self.W[x]


    def backward(self, grad_out):
        """Backward pass for word embedding layer. We cannot back-propagate into the input since 
        they are integers, so we only return gradient for the word embedding matrix W.
        
        Args:
            grad_out (np.array): Upstream gradients of shape (N, T, D)
        
        Returns:
            grad_W (np.array): Gradient of word embedding matrix, of shape (V, D)
        """
        if self.x is None:
            raise "forward pass must occur before backward pass"

        grad_W = np.zeros(self.W.shape)

        ################################################
        # The following operation is equivalent to
        # for row in range(N):
        #    for col in range(T):
        #    dW[ x[row,col]  , :] += dout[row,col, :]
        ################################################
        np.add.at(grad_W, self.x, grad_out)

        return grad_W
