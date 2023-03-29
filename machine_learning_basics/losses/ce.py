import numpy as np

class CrossEntropy:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_true, y_logits):
        shifted_logits = y_logits - np.max(y_logits, axis=1, keepdims=True)
        denominator = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(denominator)
        self.y_pred = np.exp(log_probs)
        self.y_true = y_true
        # Assume that y_true and y_logits have the same shape,
        # y_true has probabilities for multiple classes, i.e. one-hot encoding.
        # [
        #   [p1, p2, p3, ..., pC] => [0, 0, 1, ..., 0]
        #   [p1, p2, p3, ..., pC] => [0, 1, 0, ..., 0]
        # ]
        return -np.sum(y_true * log_probs) / y_true.shape[0]
    
    def gradients(self):
        if self.y_pred is None:
            raise ValueError("loss function must be forward propagated first")
        grad_y = (self.y_pred - self.y_true) / self.y_true.shape[0]
        return grad_y
