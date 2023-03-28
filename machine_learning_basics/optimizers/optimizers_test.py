import numpy as np

from optimizers.adam import Adam


def test_adam():
    adam = Adam(learning_rate=10)
    N = 4
    X = np.random.randn(N, 10)
    params = {
        "weights": np.random.randn(10, 10)        
    }

    # We need to compute the output and use its shape to infer the correct gradient update that is
    # backpropagated to weight.
    y = np.matmul(X, params["weights"])
    grad_y = np.ones(y.shape) # (N, 10) 

    # Function is f(x) = XW, and gradient of W with respect to f(x) is simply X but transposed.
    # Gradients won't change, because we have a linear function. It will descent until weights are
    # zeros.
    grads = {
        "weights": np.matmul(X.T, grad_y) 
    }
    old_weights = params["weights"]
    adam.update(1, "weights", params, grads)
    new_weights = params["weights"]
    assert np.all(new_weights - old_weights != 0)
