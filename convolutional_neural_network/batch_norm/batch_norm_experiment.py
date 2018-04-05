import numpy as np


def forward_prop(hidden_layer_sizes, weight_init_func):
    """This is a simple experiment on showing how weight initialization can impact activation through deep layers
    """
    # Extract the first hidden layer dimension
    h1_dim = hidden_layer_sizes[0]

    # Randomly initialize 1000 inputs
    inputs = np.random.randn(1000, h1_dim)

    nonlinearities = ['tanh'] * len(hidden_layer_sizes)
    act_func = {
        'relu': lambda x: np.maximum(0, x),
        'tanh': lambda x: np.tanh(x)
    }

    hidden_layer_acts = dict()

    for i in range(len(hidden_layer_sizes)):
        if i == 0:
            X = inputs
        else:
            X = hidden_layer_acts[i - 1]

        fan_in = X.shape[1]
        fan_out = hidden_layer_sizes[i]

        W = weight_init_func(fan_in, fan_out)
        H = np.dot(X, W)
        H = act_func[nonlinearities[i]](H)

        hidden_layer_acts[i] = H

    hidden_layer_means = [np.mean(H) for i, H in hidden_layer_acts.items()]
    hidden_layer_stds = [np.std(H) for i, H in hidden_layer_acts.items()]

    return hidden_layer_acts, hidden_layer_means, hidden_layer_stds
