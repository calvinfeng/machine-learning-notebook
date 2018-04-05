import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from demo_net.data_util import load_iris_data
from pdb import set_trace


class SquaredErrorLossNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
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
                training_accuracy = (np.argmax(acts['a2'], axis=1) == np.argmax(y, axis=1)).mean()
                acc_hist.append(training_accuracy)
                loss_hist.append(loss)
                iters.append(it)

            self.params['W2'] -= learning_rate * grads['W2']
            self.params['W1'] -= learning_rate * grads['W1']

        return iters, loss_hist, acc_hist

    def predict(self, x):
        act = self._forward_prop(x)
        return np.argmax(act['a2'], axis=1)

    def _forward_prop(self, x):
        acts = dict()
        acts['theta1'] = np.dot(x, self.params['W1']) # (N x input_dim)(input_dim x hidden_dim)
        acts['a1'] = 1 / (1 + np.exp(-acts['theta1'])) # (N x hidden_dim)
        acts['theta2'] = np.dot(acts['a1'], self.params['W2']) # (N x hidden_dim)(hidden_dim x output_dim)
        acts['a2'] = 1 / (1 + np.exp(-acts['theta2'])) # known as scores (N x output_dim)
        return acts

    def _loss(self, acts, y):
        return np.square(acts['a2'] - y).sum()

    def _backward_prop(self, x, y, acts):
        grads = dict()
        grads['a2'] = 2.0 * (acts['a2'] - y) # (N x output_dim)
        grads['theta2'] = grads['a2'] * ((1 - acts['a2']) * acts['a2']) # (N x output_dim)
        grads['W2'] = np.dot(acts['a1'].T, grads['theta2']) # (hidden_dim x N)(N x output_dim)
        grads['a1'] = np.dot(grads['theta2'], self.params['W2'].T) # (N x output_dim)(output_dim x hidden_dim)
        grads['theta1'] = grads['a1'] * ((1 - acts['a1']) * acts['a1']) # (N x hidden_dim)
        grads['W1'] = np.dot(x.T, grads['theta1']) # (input_dim x N)(N x hidden_dim)
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
    xtr, ytr = load_iris_data('./datasets/iris_train.csv')
    xte, yte = load_iris_data('./datasets/iris_test.csv')
    
    input_dim, hidden_dim, output_dim = xtr.shape[1], 5, ytr.shape[1]
    network = SquaredErrorLossNetwork(input_dim, hidden_dim, output_dim)
    print 'Performing gradient check: %s' % network.gradient_check(xtr, ytr)

    test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
    print 'Test accuracy before training: %s' % str(test_acc)

    iters, loss_hist, acc_hist = network.train(xtr, ytr)

    test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
    print 'Test accuracy after training: %s' % str(test_acc)

    plt.subplot(2, 1, 1)
    plt.plot(iters, loss_hist)
    plt.title('Loss vs Iterations')

    plt.subplot(2, 1, 2)
    plt.plot(iters, acc_hist)
    plt.title('Accuracy vs Iterations')

    plt.show()
