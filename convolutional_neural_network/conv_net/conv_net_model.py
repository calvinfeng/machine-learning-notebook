# Created: March, 2018
# Author(s): Calvin Feng

from composite import *
from layer import *
from solver import Solver
from composite.gradient_check import eval_numerical_gradient, rel_error
import data_utils
import numpy as np
import time
from data_utils import get_preprocessed_CIFAR10


class ConvNetModel(object):
    """ConvNetworkModel implements a convolutional network with the following architecture

    conv -> relu -> 2x2 max pool -> batch norm -> affine -> relu -> affine -> softmax

    Assuming input dimension is of the format NCHW, TODO: change data format to NHWC
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, hidden_dim=100, num_classes=10,
                weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.reg = reg
        self.dtype = dtype
        self.params = dict()
        # Manually define weights
        # Input starts with (N, 3, 32, 32)
        #   1. Convolutional layer will produce (N, 32, 32, 32) using 32 7x7 filters with stride = 1
        #   2. ReLu layer will conserve dimension
        #   3. Max pooling layer will produce (N, 32, 16, 16) using 2x2 filter with stride = 2
        #   4. Affine layer will produce (N, hidden_dim)
        #   5. Batch normlization layer will conserve dimension
        #   6. ReLU layer will conserve dimension
        #   7. Affine layer will produce (N, num_classes)
        F, C, H, W = (num_filters,) + input_dim
        self.params['W1'] = np.random.normal(0, scale=weight_scale, size=(F, C, filter_size, filter_size))
        self.params['b1'] = np.zeros((F,))

        self.params['W2'] = np.random.normal(0, scale=weight_scale, size=(F * (H // 2) * (W // 2), hidden_dim))
        self.params['b2'] = np.zeros((hidden_dim,))

        self.params['W3'] = np.random.normal(0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros((num_classes,))

        self.params['gamma'] = np.ones((1, 1))
        self.params['beta'] = np.zeros((1, 1))

        self.layer_1 = ConvReLUMaxPool(conv_param={'stride': 1, 'pad': (filter_size - 1) // 2},
                                    pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2})
        self.layer_2 = AffineBatchNormReLU(batch_norm_param={'eps': 1e-5, 'momentum': 0.95})
        self.layer_3 = Affine()

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Computes loss and gradient if label data is provided, else returns scores

        Args:
            x: Input matrix of any shape, depending on initialization values of the model
            y: Labels for training data x, of shape (N,)

        Returns:
            loss: Loss calculation of the model
            scores: Scores of each test example, of shape (N, num_classes)
            grads: Dictionary that contains gradients for each parameter
        """
        X = X.astype(self.dtype)

        mode = 'train'
        if y is None:
            mode = 'test'

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma, beta = self.params['gamma'], self.params['beta']

        Act1 = self.layer_1.forward_pass(X, W1, b1)
        Act2 = self.layer_2.forward_pass(Act1, W2, b2, gamma, beta, mode=mode)
        Act3 = self.layer_3.forward_pass(Act2, W3, b3)

        scores = Act3
        if mode == 'test':
            return scores

        loss, grad_score = self._softmax(scores, y)

        l = 1
        while l <= 3:
            weight = self.params['W' + str(l)]
            loss += 0.5 * self.reg * np.sum(weight)
            l += 1

        grads = dict()

        grad_act2, grads['W3'], grads['b3'] = self.layer_3.backward_pass(grad_score)
        grads['W3'] += self.reg * self.params['W3']

        grad_act1, grads['W2'], grads['b2'], grads['gamma'], grads['beta'] = self.layer_2.backward_pass(grad_act2)
        grads['W2'] += self.reg * self.params['W2']

        grad_x, grads['W1'], grads['b1'] = self.layer_1.backward_pass(grad_act1)
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads

    def gradient_check(self, X, y):
        """Runs gradient check on every parameter of the model

        Args:
            X: Input data in matrix form, of any shape
            y: Vector of labels
        """
        print "Running numeric gradient check with reg = %s" % self.reg

        loss, grads = self.loss(X, y)
        for param_key in sorted(self.params):
            f = lambda _: self.loss(X, y)[0]
            num_grad = eval_numerical_gradient(f, self.params[param_key], verbose=False)
            print "%s relative error: %.2e" % (param_key, rel_error(num_grad, grads[param_key]))

    def _softmax(self, x, y):
        """Computes the loss and gradient for softmax classification

        Args:
            x: Input data, of shape (N, C) where x[i, j] is the score for jth class for the ith input
            y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

        Returns:
            loss: Scalar value of the loss
            grad_x: Gradient of the loss with respect to x
        """
        # Ensure numerical stability by shifting the input matrix by its largest value in each row.
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)

        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N

        probs = np.exp(log_probs)
        grad_x = probs.copy()
        grad_x[np.arange(N), y] -= 1
        grad_x /= N

        return loss, grad_x


if __name__ == '__main__':
    # Let's get some data in first
    feed_dict = get_preprocessed_CIFAR10('datasets/cifar-10-batches-py')

    for key, value in feed_dict.iteritems():
        print "%s has shape: %s" % (key, value.shape)

    # Define the model
    model = ConvNetModel()
    t0 = time.time()
    solver = Solver(model,
                    feed_dict,
                    update_rule='sgd_momentum',
                    num_epochs=4,
                    batch_size=100,
                    optim_config={'learning_rate': 1e-3},
                    verbose=True)
    solver.train()
    tf = time.time()
