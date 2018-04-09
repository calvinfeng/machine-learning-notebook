import numpy as np


class FourLayerNeuralNetwork(object):
    """A three-layer fully-connected neural network.
    Architecture:
        input -> fully-connected -> ReLU -> fully-connected -> ReLU -> fully-connected -> softmax -> prediction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        self.params = dict()
        # Input layer
        self.params['W1'] = std * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        # First hidden layer
        self.params['W2'] = std * np.random.randn(hidden_dim, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # Second hidden layer
        self.params['W3'] = std * np.random.rand(hidden_dim, output_dim)
        self.params['b3'] = np.zeros(output_dim)

    def train(self, X, y, learning_rate=1e-5, learning_rate_decay=0.95, reg=0, num_iters=5000, batch_size=200):
        """Train my shit with stochastic gradient descent
        """
        N = X.shape[0]
        report_interval = max(N / batch_size, 1)

        # Collect records
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            I = np.random.randint(N, size=batch_size)
            X_batch = np.array(X[I])
            y_batch = np.array(y[I])

            # Compute loss and gradients using current mini-batch
            act = self._forward_prop(X_batch)
            loss = self._loss(X_batch, y_batch, act['probs'], reg)
            grads = self._gradients(X_batch, y_batch, act, reg)

            loss_history.append(loss)

            for param_name in self.params:
                self.params[param_name] -= grads[param_name]*learning_rate

            if it % report_interval == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                learning_rate *= learning_rate_decay # Decay every set interval

        return loss_history

    def predict(self, X):
        act = self._forward_prop(X)
        return np.argmax(act['probs'], axis=1)

    def gradient_check(self, X, y, reg=0.05, h=1e-5):
        # Forward prop, compute loss and gradients
        activations = self._forward_prop(X)
        loss = self._loss(X, y, activations['probs'], reg)
        grads = self._gradients(X, y, activations, reg)

        # Compute numerical gradients with slope test
        num_grads, err  = dict(), 0
        p_count = 0
        for param_name in self.params:
            # Extract one of the parameter, e.g. W1, W2, W3 etc...
            num_grads[param_name] = np.zeros_like(self.params[param_name])

            it = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                # Suppose our loss function is a f(p) and p is the param vector
                midx = it.multi_index
                p = self.params[param_name][midx]

                # Evaluate loss function at p + h
                self.params[param_name][midx] = p + h
                act = self._forward_prop(X)
                fp_plus_h = self._loss(X, y, act['probs'], reg)

                # Evaluate loss function at p - h
                self.params[param_name][midx] = p - h
                act = self._forward_prop(X)
                fp_minus_h = self._loss(X, y, act['probs'], reg)

                # Restore original value
                self.params[param_name][midx] = p

                # Slope
                num_grads[param_name][midx] = (fp_plus_h - fp_minus_h) / (2 * h)

                err += np.abs(num_grads[param_name][midx] - grads[param_name][midx])
                p_count += 1
                it.iternext()

        return err / p_count

    def _loss(self, X, y, probs, reg):
        """
        Args:
            X: Input matrix, each row represents an input vector for each example
            y: Label matrix, each row represents an classification vector for each example
            probs: Probabilities of classification for each example
            reg: Regularization strength

        Returns:
            loss: The total loss of the current model
        """
        loss = 0
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        for ith_example, k in np.ndenumerate(y):
            loss += -np.log(probs[ith_example][k])

        N = X.shape[0]
        loss = loss / N
        loss += reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        return loss

    def _gradients(self, X, y, act, reg):
        """Compute the gradients for all of the parameters within the network

        Args:
            X: Input matrix, each row represents an input vector for each example
            y: Label matrix, each row represents an classification vector for each example
            act: Activation map which contains all the activation vectors for each layer of the network
            reg: Regularization strength
        """
        N = X.shape[0]
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        # Define a gradient dictionary
        grads = dict()

        # Computing gradients of softmax score
        dscores = act['probs']
        dscores[range(N), y] -= 1
        dscores /= N # (N x O)

        # Using ReLU activation to compute gradient of W3 and b3 w.r.t. loss
        a2 = act['a2'] # Dimension: (N x H)
        grads['W3'] = np.dot(a2.T, dscores) # Dimension: (H x N)(N x O) => (H x O)
        grads['W3'] += reg*W3
        grads['b3'] = np.sum(dscores, axis=0) # Dimension: (1 x O)

        # Computing gradient of theta2 score
        da2 = np.dot(dscores, W3.T) # Dimension: (N x O)(O x H) => (N x H)
        dtheta2 = da2
        dtheta2[a2 <= 0] = 0 # Dimension: (N x H)

        # Using ReLU activation to compute gradient of W2 and b2 w.r.t loss
        a1 = act['a1'] # Dimension: (N x H)
        grads['W2'] = np.dot(a1.T, dtheta2) # Dimension: (H x N)(N x H) => (H x H)
        grads['W2'] += reg*W2
        grads['b2'] = np.sum(dtheta2, axis=0) # Dimension: (1 x H)

        # Computing gradient of theta1 score
        da1 = np.dot(dtheta2, W2.T) # Dimension: (N x H)(H x H) => (N x H)
        dtheta1 = da1
        dtheta1[a1 <= 0] = 0 # Dimension: (N x H)

        # Using ReLU activation to compute gradient of W1 and b1 w.r.t loss
        grads['W1'] = np.dot(X.T, dtheta1) # Dimension: (D x N)(N x H) = (D x H)
        grads['W1'] += reg*W1
        grads['b1'] = np.sum(dtheta1, axis=0) # Dimension: (1 x H)

        return grads

    def _forward_prop(self, X):
        """
        Args:
            X: Input matrix, each row represents an input vector for each example
            N: Number of input examples
            D: Dimension of the input vector (a.k.a input_dim)
            H: Dimension of hidden vector (a.k.a hidden_dim)
            O: Dimension of output vector (a.k.a output_dim)

        Returns:
            probs: Probabilities of classification for each example
        """
        N, D = X.shape

        # Extracting parameters, a.k.a weights
        W1, b1 = self.params['W1'], self.params['b1'] # (D x H) + D * (1 x H) *broadcasting technique vertically
        W2, b2 = self.params['W2'], self.params['b2'] # (H x H) + H * (1 x H)
        W3, b3 = self.params['W3'], self.params['b3'] # (H x O) + H * (1 x O)

        # Activations
        act = dict()

        act['theta1'] = X.dot(W1) + b1 # Multiply gate (N x D) (D x H) => (N x H)
        act['a1'] = np.maximum(act['theta1'], 0) # ReLU gate

        act['theta2'] = act['a1'].dot(W2) + b2 # Multiply gate (N x H) (H x H) => (N x H)
        act['a2'] = np.maximum(act['theta2'], 0) # ReLU gate

        act['scores'] = act['a2'].dot(W3) + b3 # Multiply gate (N x H)(H x O) => (N x O)
        act['exp_scores'] = np.exp(act['scores']) # Softmax

        act['probs'] = act['exp_scores'] / np.sum(act['exp_scores'], axis=1, keepdims=True) # Softmax => (N x O)

        return act


if __name__ == "__main__":
    from neuralnet.tests.test_forward_prop import generate_random_data
    N, input_dim, hidden_dim, output_dim = 100, 10, 10, 5
    rand_X, rand_y = generate_random_data(N, input_dim, output_dim)
    network = FourLayerNeuralNetwork(input_dim, hidden_dim, output_dim, std=0.25)
    loss_hist = network.train(rand_X, rand_y, learning_rate=5e-1, reg=0)
