import numpy as np


class AdaGradOptimizer(object):
    def __init__(self, model, learning_rate=1e-1):
        """
        :param RNNModel model: Instance of recurrent neural network model
        """
        self.learning_rate = learning_rate
        self.model = model
        # Create memory variables for Adagrad
        self.mem = dict()
        for name in model.params:
            self.mem[name] = np.zeros_like(model.params[name])

    def update_param(self, grads):
        """
        :param dict grads: Dictionary of gradients for each parameter of the model
        """
        for name in self.model.params:
            self.mem[name] += grads[name] * grads[name]
            self.model.params[name] += -1* (self.learning_rate * grads[name] / np.sqrt(self.mem[name] + 1e-8))
