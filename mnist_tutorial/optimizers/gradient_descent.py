import numpy as np


class GradientDescent(object):
    def __init__(self, learning_rate=1e-2, decay=1e-4, momentum=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum

    def update(self, w, grad_w, config=None):
        if config is None:
            config = {}
        
        v = config.get('velocity', np.zeros_like(w))
        v = self.momentum * v - self.learning_rate * grad_w
        next_w = w + v

        config['velocity'] = v

        return next_w, config

    def lr_decay(self):
        self.learning_rate *= (1.0 - self.decay)