# Created: April, 2018
# Author(s): Calvin Feng

from word_embedding_layer import WordEmbeddingLayer
from gradient_check import eval_numerical_gradient_array, rel_error
import numpy as np
import unittest


class WordEmbeddingLayerTest(unittest.TestCase):
    def test_forward(self):
        V, D = 5, 3
        
        # Create the layer
        layer = WordEmbeddingLayer(V, D)
        layer.W = np.linspace(0, 1, num=V*D).reshape(V, D)
        
        # Create some arbitrary inputs
        x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])

        out = layer.forward(x)
        
        expected = np.asarray([
            [[ 0.,          0.07142857,  0.14285714],
             [ 0.64285714,  0.71428571,  0.78571429],
             [ 0.21428571,  0.28571429,  0.35714286],
             [ 0.42857143,  0.5,         0.57142857]],
            [[ 0.42857143,  0.5,         0.57142857],
             [ 0.21428571,  0.28571429,  0.35714286],
             [ 0.,          0.07142857,  0.14285714],
             [ 0.64285714,  0.71428571,  0.78571429]]])

        self.assertAlmostEqual(rel_error(expected, out), 1e-9, places=2)

    def test_backward(self):
        np.random.seed(271)

        N, T, V, D = 50, 3, 5, 6

        # Create the layer
        layer = WordEmbeddingLayer(V, D)
        layer.W = np.random.randn(V, D)       
        
        # Create some arbitrary inputs
        x = np.random.randint(V, size=(N, T))
        
        out = layer.forward(x)

        grad_out = np.random.randn(*out.shape)

        grad_W = layer.backward(grad_out)

        f = lambda W: layer.forward(x, W=W)
        grad_W_num = eval_numerical_gradient_array(f, layer.W, grad_out)

        self.assertAlmostEqual(rel_error(grad_W_num, grad_W), 1e-9, places=2)


def main():
    from helpers import load_text

    vocab_to_idx, _ = load_text('datasets/text.txt')
    layer = WordEmbeddingLayer(len(vocab_to_idx), 128)

    input_sentence = "both that morning equally lay"

    # As usual, N is the size of our mini-batch and T is the sequencel length
    N, T = 6, 5
    x = np.array(N * [list(map(lambda word: vocab_to_idx[word], input_sentence.split()))])
    print 'x shape', x.shape
    print 'forward pass output shape', layer.forward(x).shape

    grad_out = np.random.randn(N, T, layer.D)
    print 'gradient shape', layer.backward(grad_out).shape


if __name__ == "__main__":
    main()