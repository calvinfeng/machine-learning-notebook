from layers import Dense
import numpy as np


# Input dimension - 10
# Final output dimension - 1 
layer_dimens = [(10, 20), (20, 50), (50, 100), (100, 5), (5, 1)]

# Declare weights and biases for every layer
weights = dict()
biases = dict()

for layer_num in range(len(layer_dimens)):
    # De-constructor dimension into in and out
    in_dim, out_dim = layer_dimens[layer_num]

    # Randomly initialize some weights
    weights[layer_num] = np.random.randn(in_dim, out_dim)
    biases[layer_num] = np.zeros(out_dim)

layers = dict()
# TODO: Create # of layers depending on # of dimensions

# Use forward in each layer to compute an output.

