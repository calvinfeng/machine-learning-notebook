import numpy as np
from helpers import *

class WordEmbeddingLayer(object):
    def __init__(self, vocab_size, word_vec_dim):
        """
        :param int vocab_size: Number of words in the dictionary.
        :param int word_vec_dim: Dimension of the desired word vector.
        """
        self.V = vocab_size
        self.D = word_vec_dim
        self.W = np.random.randn(self.V, self.D)
        self.W /= 100
        self.x = None

    def forward(self, x):
        """Forward pass for word embedding layer. This function operates on mini-batches of size N where each sequence
        has length T.

        :param np.array x: Integer array of shape (N, T), each element idx of x must be in the range 0 <= idx < V
        :return np.array: Array of shape (N, T, D) which represents the word vectors
        """
        self.x = x
        return self.W[x]


    def backward(self, grad_out):
        """Backward pass for word embedding layer. We cannot back-propagate into the input since they are integers, so
        we only return gradient for the word embedding matrix W.

        :param np.array grad_out: Upstream gradients of shape (N, T, D)
        :return np.array grad_W: Gradient of word embedding matrix, of shape (V, D)
        """
        if x is None:
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


if __name__ == "__main__":
    vocab_to_idx, idx_to_vocab = load_text('datasets/text.txt')
    layer = WordEmbeddingLayer(len(vocab_to_idx), 128)

    input_sentence = "both that morning equally lay"

    # As usual, N is the size of our mini-batch and T is the sequencel length
    N, T = 6, 5
    x = np.array(N * [list(map(lambda word: vocab_to_idx[word], input_sentence.split()))])
    print 'x shape', x.shape
    print 'forward pass output shape', layer.forward(x).shape

    grad_out = np.random.randn(N, T, layer.D)
    print 'gradient shape', layer.backward(grad_out).shape
