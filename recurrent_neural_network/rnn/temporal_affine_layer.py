# Created: April, 2018
# Author(s): Calvin Feng

import numpy as np


class TemporalAffineLayer(object):
    def __init__(self, H, M):
        """
        Args:
            H (int): Dimension of hidden state from LSTM cell.
            M (int): Number of words in the vocabulary.
        """
        self.W = np.random.randn(H, M)
        self.W /= np.sqrt(H)
        self.b = np.zeros(M)
        self.x = None

    def forward(self, x, W=None, b=None):
        """Forward pass for temporarl affine layer. The input is a set of H-dimensional vectors 
        arranged into a mini-batch of N timeseries, each of length T. We use an affine function to 
        map each of these vectors into a new vector of dimension M. This is equivalent to mapping
        hidden state representation into word classification representation.
        
        Args:
            x (np.array): Input data of shape (N, T, H)
            W (np.array): Optional input weights of shape (H, M)
            b (np.array): Optional biases of shape (M,)
        """
        if W is not None and b is not None:
            self.W = W
            self.b = b

        N, T, H = x.shape
        M = self.b.shape[0]

        self.x = x
        return np.dot(x.reshape(N * T, H), self.W).reshape(N, T, M) + self.b   
        
    def backward(self, grad_out):
        """
        Args:
            grad_out (np.array): Upstream gradients of shape (N, T, M)
        
        Returns tuple:
            - grad_x (np.array): Gradients of input, of shape (N, T, H)
            - grad_W (np.array): Gradients of weights, of shape (H, M)
            - grad_b (np.array): Gradients of biases, of shape (M,)
        """
        N, T, H = self.x.shape
        M = self.b.shape[0]

        grad_x = np.dot(grad_out.reshape(N*T, M), self.W.T).reshape(N, T, H)
        grad_W = np.dot(grad_out.reshape(N*T, M).T, self.x.reshape(N*T, H)).T
        grad_b = grad_out.sum(axis=(0, 1))

        return grad_x, grad_W, grad_b