# Created: April, 2018
# Author(s): Calvin Feng

from word_embedding_layer import WordEmbeddingLayer
from lstm_recurrent_layer import LSTMRecurrentLayer
from temporal_affine_layer import TemporalAffineLayer
from temporal_softmax import temporal_softmax_loss
from data_util import load_random_sentences
import numpy as np


class LSTMRecurrentModel(object):
    def __init__(self, word_to_idx, W=128, H=128):
        """
        Args:
            word_to_idx (dict): A dictionary giving the vocabulary. It contains V entries, and maps
                                maps each string to a unique integer in the range [0, V).
            W (int): Dimension of word vector.
            H (int): Dimension of hidden state of the RNN layer.
        """
        self.word_to_idx = word_to_idx
        self.W = W 
        self.H = H
        
        self._null_token = word_to_idx['<NULL>']
        self._start_token = word_to_idx.get('<START>', None)
        self._end_token = word_to_idx.get('<END>', None)

        # Initialize layers
        self.word_embedding_layer = WordEmbeddingLayer(len(word_to_idx), W) 
        self.lstm_recurrent_layer = LSTMRecurrentLayer(W, H)
        self.temporal_affine_layer = TemporalAffineLayer(H, len(word_to_idx))

        # If we were to use feature vector coming from a CNN, then we will need the following:
        ###########################################################################################
        # D represents feature dimension of each image coming from a CNN (before feeding 
        # into FC -> Softmax.)
        # 
        # W_proj = np.random.randn(D, H) / np.sqrt(D)
        # b_proj = np.zeros(H)
        # h0 = np.dot(features, W_proj) + b_proj  ===> (N, D)(D, H) + (D,)  ===> (N, H)
        # 
        # We can treat the above as another affine layer.
        ###########################################################################################        

    def loss(self, sentences):
        """Compute training-time loss for the model.

        Args:
            sentences (np.array): Ground-truth sentences or captions; an integer array of shape (N, T) 
                                  where each element is in the range 0 <= y[n, t] < V.
        
        Returns tuple:
            loss (float): Scalar loss.
            grads (dict): Dictionary of gradients
        """
        # Cut sentences into two pieces, sentences_in has everything but the last word and will be
        # input to the RNN; sentences_out has everything but teh first word and this is what we 
        # will expefct the RNN to generate. RNN should produce word (t + 1) after receiving word t.
        # The first element of sentences_in will be the START token and the first element of 
        # sentences_out will be the first word.
        N, _ = sentences.shape
        sentences_in = sentences[:, :-1]
        sentences_out = sentences[:, 1:]
        
        mask = (sentences_out != self._null_token)
        h0 = np.zeros((N, self.H))

        word_embedding_out = self.word_embedding_layer.forward(sentences_in)
        lstm_out = self.lstm_recurrent_layer.forward(word_embedding_out, h0)
        affine_out = self.temporal_affine_layer.forward(lstm_out)
        
        # Affine out is the score for softmax classification
        score = affine_out
        loss, grad_score = temporal_softmax_loss(score, sentences_out, mask, verbose=False)
        
        grads = dict()
        grads['temporal_affine'] = self.temporal_affine_layer.backward(grad_score)
        grads['lstm'] = self.lstm_recurrent_layer.backward(grads['temporal_affine'][0])
        grads['word_embedding'] = self.word_embedding_layer.backward(grads['lstm'][0])

        return score, loss, grads


def main():
    sentences, word_to_idx, _ = load_random_sentences('datasets/random_sentences.txt', 20)
    model = LSTMRecurrentModel(word_to_idx)    
    loss, _, _ = model.loss(sentences)
    print loss


if __name__ == "__main__":
    main()