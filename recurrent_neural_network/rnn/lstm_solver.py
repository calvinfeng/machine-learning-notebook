import numpy as np
import optimizers
import matplotlib.pyplot as plt
from data_util import load_random_sentences
from lstm_recurrent_model import LSTMRecurrentModel


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
        self.learning_rate_decay = kwargs.pop('learning_rate_decay', 1.0)
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
        """Reset book-keeping variables for optimization
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Initialize optimization configuration for each layer
        self.optim_configs = dict() 
        for param in ['word_embedding', 'lstm', 'temporal_affine']:
            self.optim_configs[param] = dict()

    def _step(self):
        """Make a single gradient update
        """
        num_train = self.data['training'].shape[0]

        # Create a mini-batch of training data
        batch_mask = np.random.choice(num_train, self.batch_size)
        minibatch = self.data['training'][batch_mask]

        # Compute loss and gradients
        _, loss, grads = self.model.loss(minibatch)
        self.loss_history.append(loss)

        self.model.temporal_affine_layer.update(grads['temporal_affine'], 
                                                self.update_rule, 
                                                self.optim_configs['temporal_affine'])
        self.model.lstm_recurrent_layer.update(grads['lstm'], 
                                               self.update_rule,
                                               self.optim_configs['lstm'])
        self.model.word_embedding_layer.update(grads['word_embedding'], 
                                               self.update_rule,
                                               self.optim_configs['word_embedding'])

    def train(self):
        num_train = self.data['training'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        total_iterations = self.num_epochs * iterations_per_epoch

        for t in range(total_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print '(Iteration %d / %d): loss: %f' % (t + 1, total_iterations, self.loss_history[-1])

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

                # Apply decay on learning rate
                for layer in self.optim_configs:
                    for param in self.optim_configs[layer]:
                        self.optim_configs[layer][param]['learning_rate'] *= self.learning_rate_decay
                
                # Check model accuracy
                sentences_in = self.data['training'][:, :-1]
                sentences_out = self.data['training'][:, 1:]
                # print self.check_accuracy(sentences_in, sentences_out)
                self.model.sample(sentences_in)

        if self.verbose:
            plt.plot(np.arange(total_iterations), self.loss_history)
            plt.show()

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """Check accuracy of the model on the provided data

        Args:
            X (np.array): Array of data, of shape (N, d_1, ..., d_k)
            y (np.array): Array of labels, of shape (N,)
            num_samples (int): If not None, subsample the data and only test the model on
                               num_samples datapoints.
            batch_size (int): Split X and y into batches of this size to avoid using too much memory

        Returns:
            acc (float): Scalar giving the fraction of instances that were correctly classified by
                         the model.
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            score, _, _ = self.model.loss(X[start:end])
            y_pred.append(np.argmax(score, axis=2))
            
        y_pred = np.hstack(y_pred)
        return np.mean(y_pred == y[:, 1:])
        

def main():
    sentences, word_to_idx, idx_to_word = load_random_sentences('datasets/random_sentences.txt', 30)
    model = LSTMRecurrentModel(word_to_idx, idx_to_word)        
    solver = LSTMSolver(model, {'training': sentences}, batch_size=20, 
                                                        num_epochs=1000, 
                                                        print_every=10,
                                                        learning_rate_decay=0.99,
                                                        update_rule='adam')
    solver.train()


if __name__ == "__main__":
    main()