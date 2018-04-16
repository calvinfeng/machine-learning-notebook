from rnn.word_embedding import WordEmbeddingLayer
from helpers import load_text
import numpy as np


def main():
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