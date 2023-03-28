import numpy as np

class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.means = {} # This is 1st moment, which estimate the mean
        self.variances = {} # This is 2nd moment, which estimate the uncentered variance

    def update(self, i, key, params, grads):
        if i < 1:
            raise ValueError("time step i should start at 1")

        if key not in self.means:
            self.means[key] = np.zeros(params[key].shape)
        
        if key not in self.variances:
            self.variances[key] = np.zeros(params[key].shape)

        prev_mean = self.means[key]
        prev_var = self.variances[key]

        self.means[key] = self.beta1 * prev_mean + (1 - self.beta1) * grads[key]
        self.variances[key] = self.beta2 * prev_var + (1 - self.beta2) * grads[key]**2

        # Bias corrected
        corrected_mean = self.means[key] / (1 - self.beta1**i)
        corrected_var = self.variances[key] / (1 - self.beta2**i)
        
        # Gradient updates
        params[key] = params[key] - self.learning_rate * corrected_mean / (np.sqrt(corrected_var) + self.epsilon)
