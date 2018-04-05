import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from demo_net.data_util import load_iris_data
from pdb import set_trace


class SoftmaxLossNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-5):
        self.params = dict()
        self.params['W1'] = std * randn(input_dim, hidden_dim) # random normal distributed
        self.params['W2'] = std * randn(hidden_dim, output_dim) # random normal distributed

    def train(self, x, y, learning_rate=1e-2, num_iters=2000, epoch=25):
        iters, loss_hist, acc_hist = [], [], []
        for it in xrange(num_iters):
            acts = self._forward_prop(x)
            loss = self._loss(acts, y)
            grads = self._backward_prop(x, y, acts)

            if it % epoch == 0:
                training_accuracy = (np.argmax(acts['a2'], axis=1) == y).mean()
                acc_hist.append(training_accuracy)
                loss_hist.append(loss)
                iters.append(it)
                learning_rate *= 0.95

            self.params['W2'] -= learning_rate * grads['W2']
            self.params['W1'] -= learning_rate * grads['W1']

        return iters, loss_hist, acc_hist

    def _forward_prop(self, x):
        acts = dict()

        # (N x input_dim)(input_dim x hidden_dim)
        acts['theta1'] = np.dot(x, self.params['W1'])

        # (N x hidden_dim)
        acts['a1'] = 1 / (1 + np.exp(-acts['theta1']))

        # (N x hidden_dim)(hidden_dim x output_dim)
        acts['theta2'] = np.dot(acts['a1'], self.params['W2'])

        # Exponential scores can become very big numbers, in order to ensure numerical stability, we need to apply the
        # following mathematical trick (shifting by max activation score.)
        shifted_logits = acts['theta2'] - np.max(acts['theta2'], axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)

        log_probs = shifted_logits - np.log(Z) # Just to avoid taking log of zero(s) by using a simple log trick.
        acts['a2'] = np.exp(log_probs)
        # acts['a2'] = np.exp(acts['theta2'])
        # acts['a2'] = acts['a2'] / np.sum(acts['a2'], axis=1, keepdims=True)

        return acts

    def _loss(self, acts, y):
        loss = 0
        N = len(y)
        for idx, val in enumerate(y):
            # val is the classification 0, 1, 2 etc...
            # idx is the ith example we are classifying
            prob = acts['a2'][idx][val]
            loss += -1 * np.log(max(prob, 1e-100)) # Prevent taking log(0) which gives infinity

        return loss / N

    def _backward_prop(self, x, y, acts):
        grads = dict()

        # Theta 2 is equivalent to the score that we propagate toward our softmax layer. We taking derivative of the score with respect to
        # cross entropy loss here
        N = len(y)
        grads['theta2'] = acts['a2']
        grads['theta2'][range(N), y] -= 1
        grads['theta2'] /= N

        # (hidden_dim x N)(N x output_dim)
        grads['W2'] = np.dot(acts['a1'].T, grads['theta2'])

        # (N x output_dim)(output_dim x hidden_dim)
        grads['a1'] = np.dot(grads['theta2'], self.params['W2'].T)

        # (N x hidden_dim)
        grads['theta1'] = grads['a1'] * ((1 - acts['a1']) * acts['a1'])

        # (input_dim x N)(N x hidden_dim)
        grads['W1'] = np.dot(x.T, grads['theta1'])

        return grads

    def _num_grads(self, x, y, h=1e-5):
        num_grads = dict()
        for param_name in self.params:
            num_grads[param_name] = np.zeros_like(self.params[param_name])

            it = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                # Suppose our loss function is a f(p) and p is the param vector
                midx = it.multi_index
                p = self.params[param_name][midx]

                # Evaluate loss function at p + h
                self.params[param_name][midx] = p + h
                acts = self._forward_prop(x)
                fp_plus_h = self._loss(acts, y)

                # Evaluate loss function at p - h
                self.params[param_name][midx] = p - h
                acts = self._forward_prop(x)
                fp_minus_h = self._loss(acts, y)

                # Restore original value
                self.params[param_name][midx] = p

                # Slope
                num_grads[param_name][midx] = (fp_plus_h - fp_minus_h) / (2 * h)

                it.iternext()

        return num_grads

    def gradient_check(self, x, y, h=1e-5):
        acts = self._forward_prop(x)
        grads = self._backward_prop(x, y, acts)
        num_grads = self._num_grads(x, y, h)
        err, param_count = 0, 0

        for param_name in self.params:
            it = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                midx = it.multi_index
                err += (np.abs(num_grads[param_name][midx] - grads[param_name][midx]) /
                    max(np.abs(num_grads[param_name][midx]), np.abs(grads[param_name][midx])))
                param_count += 1
                it.iternext()

        return err / param_count


if __name__ == "__main__":
    xtr, ytr = load_iris_data('./datasets/iris_train.csv', multi_dimen_labels=False)
    xte, yte = load_iris_data('./datasets/iris_test.csv',  multi_dimen_labels=False)
    input_dim, hidden_dim, output_dim = xtr.shape[1], 5, 3
    network = SoftmaxLossNetwork(input_dim, hidden_dim, output_dim)
    print 'Performing gradient check %s' % network.gradient_check(xtr, ytr)

    iters, loss_hist, acc_hist = network.train(xtr, ytr, learning_rate=1e-1)

    plt.subplot(2, 1, 1)
    plt.plot(iters, loss_hist)
    plt.title('Loss vs Iterations')

    print loss_hist[0]
    print loss_hist[len(loss_hist) - 1]

    plt.subplot(2, 1, 2)
    plt.plot(iters, acc_hist)
    plt.title('Accuracy vs Iterations')

    plt.show()
