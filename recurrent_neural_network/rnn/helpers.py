import numpy as np


def load_text(filepath):
    idx = 0
    vocab_to_idx, idx_to_vocab = dict(), dict()
    with open(filepath, 'r') as file:
        for line in file:
            for word in line.split():
                if vocab_to_idx.get(word, None) is None and idx_to_vocab.get(idx, None) is None:
                    vocab_to_idx[word] = idx
                    idx_to_vocab[idx] = word
                    idx += 1

    print "Number of unique words extracted from data: %d" % len(vocab_to_idx)
    return vocab_to_idx, idx_to_vocab


def sigmoid(x):
    """Numerically stable version of the logistic sigmoid function"""
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)
