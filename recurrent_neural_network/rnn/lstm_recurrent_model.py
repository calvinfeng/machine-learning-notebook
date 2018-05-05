# Created: April, 2018
# Author(s): Calvin Feng

from word_embedding_layer import WordEmbeddingLayer
from lstm_recurrent_layer import LSTMRecurrentLayer
from temporal_affine_layer import TemporalAffineLayer
from temporal_softmax import temporal_softmax_loss
from data_util import load_word_based_text_input
from data_util import START_TOKEN, END_TOKEN, NULL_TOKEN
import numpy as np


class LSTMRecurrentModel(object):
    def __init__(self, word_to_idx, idx_to_word, W=128, H=128):
        """
        Args:
            word_to_idx (dict): A dictionary giving the vocabulary. It contains V entries, and maps
                                maps each string to a unique integer in the range [0, V).
            W (int): Dimension of word vector.
            H (int): Dimension of hidden state of the RNN layer.
        """
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
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

    def loss(self, inputs, outputs):
        """Compute training-time loss for the model.

        Args:
            inputs (np.array): Input sentences or captions; an integer array of shape (N, T) 
                               where each element is in the range 0 <= y[n, t] < V.
            outputs (np.array): Expected output sentences or captions; an integer array of shape
                                (N, T) where each element is in range 0 <= y[n, t] < V.
        
        Returns tuple:
            loss (float): Scalar loss.
            grads (dict): Dictionary of gradients
        """
        N, _ = inputs.shape
        
        # Perform forward propagation
        word_embedding_out = self.word_embedding_layer.forward(inputs)
        h0 = np.zeros((N, self.H))
        lstm_out = self.lstm_recurrent_layer.forward(word_embedding_out, h0)
        affine_out = self.temporal_affine_layer.forward(lstm_out)
        
        # Affine out is the score for softmax classifi
        score = affine_out
        
        # Compute loss
        mask = (outputs != self._null_token)
        loss, grad_score = temporal_softmax_loss(score, outputs, mask, verbose=False)
        
        # Perform back propagation
        grads = dict()
        grads['temporal_affine'] = self.temporal_affine_layer.backward(grad_score)
        grads['lstm'] = self.lstm_recurrent_layer.backward(grads['temporal_affine'][0])
        grads['word_embedding'] = self.word_embedding_layer.backward(grads['lstm'][0])

        return score, loss, grads

    def sample(self, inputs, max_length=30):
        """Run a test-time forward pass for the model
        
        At each time step, we embed the current word. We pass the word vector and the previous 
        hidden state to get the next hidden state. Then use the hidden state to get scores for all
        vocab words and randomly sample a word using the softmax probabilities.

        Args:
            max_length (int): Maximum length T of the gesentencesnerated sentence.

        Returns:
            sentence (np.array): Array of shape (1, max_length) giving sampled sentence. 
        """
        N, _ = inputs.shape    
        word_embedding_out = self.word_embedding_layer.forward(inputs)
        h0 = np.zeros((N, self.H))
        lstm_out = self.lstm_recurrent_layer.forward(word_embedding_out, h0)
        affine_out = self.temporal_affine_layer.forward(lstm_out)

        score = affine_out

        N, T, vocab_size = score.shape 
        for i in range(N):
            question = self._word_vector_to_string(inputs[i])

            sampled_word_vector = []
            for t in range(T):
                prob = np.exp(score[i, t]) / np.sum(np.exp(score[i, t]))
                sampled_word_vector.append(np.random.choice(range(vocab_size), p=prob))

            answer = self._word_vector_to_string(np.array(sampled_word_vector))

            print '=============================='
            print question
            print answer
            print '=============================='
        
    def _word_vector_to_string(self, vector):
        T, = vector.shape

        words = []
        for t in range(T):
            words.append(self.idx_to_word[vector[t]])

        # Find the beginning and ending of the sentence
        start_idx, end_idx = None, None
        for idx, word in enumerate(words):
            if word == START_TOKEN and start_idx is None:
                start_idx = idx
        
            if word == END_TOKEN and end_idx is None:
                end_idx = idx
    
        if start_idx is not None and end_idx is not None:
            return ' '.join(words[start_idx+1:end_idx])
        else:
            return 'bad sentence'


def main():
    input_filepath = 'datasets/questions.txt'
    output_filepath = 'datasets/answers.txt'
    questions, answers, word_to_idx, idx_to_word = load_word_based_text_input(input_filepath, 
                                                                              output_filepath, 
                                                                              30)
    model = LSTMRecurrentModel(word_to_idx, idx_to_word)    
    _, loss, _ = model.loss(questions, answers)
    print loss


if __name__ == "__main__":
    main()