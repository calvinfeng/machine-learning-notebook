import numpy as np


class ThreeLayerNeuralNetwork(object):
    """This network exists for debugging purposes
    """
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = std * np.random.rand(hidden_dim, output_dim)
        self.params['b2'] = np.zeros(output_dim)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # First activation vector
        a1 = X.dot(W1) + b1
        relu_activation = np.maximum(a1, 0)
        # Second activation vector
        scores = relu_activation.dot(W2) + b2

        if y is None:
            return scores

        # Compute the loss using Softmax L2 loss
        loss = 0
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores, axis=1)

        for index, y_i in np.ndenumerate(y):
            # y_i is the classification for each input, says y_i = 0 which means the classification is class #1
            loss += -np.log(np.exp(scores[index][y_i])/sum_exp_scores[index])

        loss = loss / N
        loss += reg*np.sum(W1*W1)
        loss += reg*np.sum(W2*W2)

        # Backprop for Gradients
        grads = {}
        softmax_output = np.exp(scores) / np.sum(np.exp(scores),axis=1)[:,None]
        dscores = softmax_output
        dscores[range(N), y] -= 1
        dscores /= N

        grads['W2'] = relu_activation.T.dot(dscores)
        grads['b2'] = np.sum(dscores, axis = 0)
        grads['W2'] += reg*W2*2

        dhidden = np.dot(dscores, W2.T)
        dhidden[relu_activation <= 0] = 0

        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis = 0)
        grads['W1'] += reg*W1*2

        return loss, grads

    def gradient_check(self, X, y, reg=0.05, h=1e-5):
        # Forward prop, compute loss and gradients
        loss, grads = self.loss(X, y, reg)

        # Compute numerical gradients with slope test
        num_grads, err  = dict(), 0
        p_count = 0
        for param_name in self.params:
            # Extract one of the parameter, e.g. W1, W2 etc...
            num_grads[param_name] = np.zeros_like(self.params[param_name])

            it = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                # Suppose our loss function is a f(p) and p is the param vector
                midx = it.multi_index
                p = self.params[param_name][midx]

                # Evaluate loss function at p + h
                self.params[param_name][midx] = p + h
                fp_plus_h, _ = self.loss(X, y, reg)

                # Evaluate loss function at p - h
                self.params[param_name][midx] = p - h
                fp_minus_h, _ = self.loss(X, y, reg)

                # Restore original value
                self.params[param_name][midx] = p

                # Slope
                num_grads[param_name][midx] = (fp_plus_h - fp_minus_h) / (2 * h)

                err += np.abs(num_grads[param_name][midx] - grads[param_name][midx])
                p_count += 1
                it.iternext()

        return err / p_count

    def train(self, X, y, learning_rate=1e-2, learning_rate_decay=0.95, reg=5e-6, num_iters=3000, batch_size=200, verbose=True):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            I = np.random.randint(num_train, size=batch_size)
            X_batch = np.array(X[I])
            y_batch = np.array(y[I])

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            for para_name in self.params:
                self.params[para_name] -= grads[para_name]*learning_rate*learning_rate_decay

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

if __name__ == "__main__":
    from neural_network.tests.test_forward_prop import generate_random_data

    N, input_dim, hidden_dim, output_dim = 5, 10, 10, 5
    rand_X, rand_y = generate_random_data(N, input_dim, output_dim)
    network = ThreeLayerNeuralNetwork(input_dim, hidden_dim, output_dim, std=0.25)

    print network.gradient_check(rand_X, rand_y, reg=0)
    network.train(rand_X, rand_y)
