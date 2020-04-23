import torch
import torch.nn as nn
import torch.nn.functional as f


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Instantiates a neural network with the following attributes.

        :param input_dim: Number of features
        :param hidden_dim: Dimension of the hidden layer(s)
        :param output_dim: Number of outputs
        """
        super(SimpleNet, self).__init__()
        self.fully_connected_1 = nn.Linear(input_dim, hidden_dim)
        self.fully_connected_2 = nn.Linear(hidden_dim, output_dim)
        self.drop_out = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """Forward propagate the input and returns an output

        :param X: Numpy array of features of shape (num_inputs, inputs_dim)
        
        :return: A single sigmoid activated value
        """
        out = f.relu(self.fully_connected_1(X))
        out = self.drop_out(out)
        out = self.fully_connected_2(out)
        return self.sigmoid(out)
