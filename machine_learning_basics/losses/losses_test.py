import numpy as np

from losses.ce import CrossEntropy
from losses.mse import MeanSquaredError
from gradient.utils import numerical_gradient


def test_cross_entropy():
    ce = CrossEntropy()
    y_true = np.identity(5)
    y_pred = np.random.randn(5, 5)
    ce(y_true, y_pred)

    num_grad_y = numerical_gradient(lambda y: ce(y_true, y), y_pred, 1)
    grad_y = ce.gradients()

    assert np.allclose(grad_y, num_grad_y)


def test_mean_squared_error():
    mse = MeanSquaredError()
    y_true = np.random.randn(5, 5)
    y_pred = np.random.randn(5, 5)
    mse(y_true, y_pred)

    num_grad_y = numerical_gradient(lambda y: mse(y_true, y), y_pred, np.ones(y_pred.shape))
    grad_y = mse.gradients()

    assert np.allclose(grad_y, num_grad_y)
