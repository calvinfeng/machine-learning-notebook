import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.sum((y_true - y_pred)**2) / np.prod(y_pred.shape)

    def gradients(self):
        if self.y_pred is None:
            raise ValueError("loss function must be forward propagated first")

        grad_y = -1 * 2 * (self.y_true - self.y_pred)
        return grad_y
