import numpy as np

np.random.seed(1234)

from layers.dense import Dense
from layers.conv import Conv2D
from layers.relu import ReLU
from layers.sigmoid import Sigmoid
from gradient.utils import numerical_gradient


def test_dense():
    dense = Dense()
    x = np.random.randn(10, 4)
    w = np.random.randn(4, 8)
    b = np.random.randn(
        8,
    )
    y = dense(x, w, b)

    grad_out = np.ones(y.shape)
    grad_x, grad_w, grad_b = dense.gradients(grad_out)

    num_grad_x = numerical_gradient(lambda x: dense(x, w, b), x, grad_out)
    num_grad_w = numerical_gradient(lambda w: dense(x, w, b), w, grad_out)
    num_grad_b = numerical_gradient(lambda b: dense(x, w, b), b, grad_out)

    assert np.allclose(grad_x, num_grad_x)
    assert np.allclose(grad_w, num_grad_w)
    assert np.allclose(grad_b, num_grad_b)


def test_conv():
    conv = Conv2D(stride=1, padding=1)
    x = np.random.randn(1, 3, 5, 5) # 1 Sample, 3 Channels, 3x3 Image
    w = np.random.randn(1, 3, 3, 3) # 1 Filter, 3 Channels, 3x3 Kernel
    b = np.random.randn(1,)
    y = conv(x, w, b)

    grad_out = np.ones(y.shape)
    grad_x, grad_w, grad_b = conv.gradients(grad_out)

    num_grad_x = numerical_gradient(lambda x: conv(x, w, b), x, grad_out)
    num_grad_w = numerical_gradient(lambda w: conv(x, w, b), w, grad_out)
    num_grad_b = numerical_gradient(lambda b: conv(x, w, b), b, grad_out)

    assert np.allclose(grad_x, num_grad_x)
    assert np.allclose(grad_w, num_grad_w)
    assert np.allclose(grad_b, num_grad_b)


def test_relu():
    relu = ReLU()
    x = np.random.randn(10, 4)
    y = relu(x)

    grad_out = np.ones(y.shape)
    grad_x = relu.gradients(grad_out)
    num_grad_x = numerical_gradient(lambda x: relu(x), x, grad_out)

    assert np.allclose(grad_x, num_grad_x)


def test_sigmoid():
    sigmoid = Sigmoid()
    x = np.random.randn(10, 4)
    y = sigmoid(x)

    grad_out = np.ones(y.shape)
    grad_x = sigmoid.gradients(grad_out)
    num_grad_x = numerical_gradient(lambda x: sigmoid(x), x, grad_out)

    assert np.allclose(grad_x, num_grad_x)
