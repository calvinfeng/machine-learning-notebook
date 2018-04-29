import numpy as np
from data_util import load_random_sentences
from lstm_recurrent_model import LSTMRecurrentModel
import optimizers


class LSTMSolver(object):
    """LSTMSolver performs training on a LSTM recurrent neural network model.

    For the sake of simplicity and considering that this Solver isn't aimed to be applied to many
    different types of model. Solver works on a model object that must conform to the following 
    rigi API:

    - model.word_embedding_layer must be a WordEmbeddingLayer object.
    - model.lstm_recurrent_layer must be a LSTMRecurrentLayer object.
    - model.temporal_affine_layer must be a TemporalAffineLayer object.
    - model.loss(sentences) must be a function that computes training-time loss and gradients.
    """
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.learning_rate_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            raise ValueError('Unrecognized arguments %s' % list(kwargs.keys()))

        if not hasattr(optimizers, self.update_rule):
            raise ValueError('Invalid update rule %s' % self.update_rule)
        
        self.update_rule = getattr(optimizers, self.update_rule)
        self._reset()

    def _reset(self):
        pass



def main():
    sentences, word_to_idx, _ = load_random_sentences('datasets/random_sentences.txt', 20)
    model = LSTMRecurrentModel(word_to_idx)        
    solver = LSTMSolver(model, {})    


if __name__ == "__main__":
    main()