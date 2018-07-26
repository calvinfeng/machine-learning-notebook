from layers import Dense, ReLU, Softmax
from models.loss import categorical_cross_entropy
from keras.datasets import mnist
from keras.utils import to_categorical
from optimizers import GradientDescent
from exceptions import Exception
import numpy as np


class Sequential(object):
    def __init__(self):
        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.activations = {}
            
    def add(self, layer, act, layer_dim, weight_scale=1e-2):
        """Add layer and activation to the current model  
        """
        l = len(self.layers)
        self.weights[l] = np.random.normal(loc=0, scale=weight_scale, size=layer_dim)
        self.biases[l] = np.zeros(layer_dim[1],)
        self.layers[l] = layer
        self.activations[l] = act

    def loss(self, x, y):
        """Computes loss and gradients of the model

        Args:
            x (np.ndarray): Tensor of input images, of shape (N, H*W). For MNIST data, it is (N, 784)
            y (np.ndarray): Tensor of labels in one-hot representation, of shape (N, 10)
        """
        if self.loss_func is None:
            raise Exception("please provide a loss function using compile() before calling loss()")

        for i in range(len(self.layers)):
            w = self.weights[i]
            b = self.biases[i]
            x = self.layers[i].forward_pass(x, w, b)
            x = self.activations[i].forward_pass(x)
        
        y_pred = x
        loss = self.loss_func(y_pred, y)
        grad_weights, grad_biases = {}, {}
        
        grad_upstream = y
        for i in reversed(range(len(self.layers))):
            grad_upstream = self.activations[i].backward_pass(grad_upstream)
            grad_upstream, grad_w, grad_b = self.layers[i].backward_pass(grad_upstream)
            grad_weights[i] = grad_w
            grad_biases[i] = grad_b

        return loss, grad_weights, grad_biases, y_pred

    def compile(self, optimizer=None, loss_func=None):
        self.optimizer = optimizer
        self.optim_config = {}
        self.loss_func = loss_func

    def fit(self, x, y, epochs=10, batch_size=50, verbose=False):
        N = x.shape[0]
        iters_per_epoch = max(N // batch_size, 1)
        num_iters = epochs * iters_per_epoch
        
        epoch = 1
        loss_history = []
        acc_history = []
        for i in range(num_iters):
            batch_mask = np.random.choice(N, batch_size)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]
            loss_history.append(self._training_step(x_batch, y_batch))

            if verbose:
                print "Iteration (%d/%d) loss: %f" % (i+1, num_iters, loss_history[-1])

            epoch_end = (i+1) % iters_per_epoch == 0
            if epoch_end:
                epoch += 1
                self.optimizer.lr_decay()
            
            if i == 0 or i == num_iters - 1 or epoch_end:
                loss, acc = self.evaluate(x_batch, y_batch)
                print "Epoch (%d/%d) training accuracy: %f and training loss %f" % (epoch, epochs, acc, loss)
                acc_history.append(acc)
                
    def evaluate(self, x, y, batch_size=50):
        N = x.shape[0]
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1

        y_pred = None
        for i in range(num_batches):
            start = i * batch_size
            end = (i+1) * batch_size
            loss, _, _, probs = self.loss(x[start:end], y[start:end])

            if y_pred is None:
                y_pred = probs
            else:
                y_pred = np.concatenate(y_pred, probs) 
        
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

        return loss, acc
                
    def _training_step(self, x, y):
        loss, grad_weights, grad_biases, _ = self.loss(x, y)
        for i in grad_weights:
            config = self.optim_config.get('w'+str(i))
            next_w, config = self.optimizer.update(self.weights[i], grad_weights[i], config)
            self.weights[i] = next_w
            self.optim_config['w'+str(i)] = config
        
        for i in grad_biases:
            config = self.optim_config.get('b'+str(i))
            next_b, config = self.optimizer.update(self.biases[i], grad_biases[i])
            self.biases[i] = next_b
            self.optim_config['b'+str(i)] = config

        return loss