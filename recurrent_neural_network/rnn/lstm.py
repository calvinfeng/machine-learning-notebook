import numpy as np
from helpers import *


class LSTMLayer(object):
    def __init__(self, word_vec_dim, hidden_dim):
        """Initialize a long short-term memory recurrent layer

        The factor of 4 is needed for i, f, o, g gates. Instead of declaring individual matrix for each of them, we
        can combine all four of them into one.

        :param int wordvec_dim: Dimension of the word vector
        :param int hidden_dim: Dimension for the hidden layer
        """
        self.Wx = np.random.randn(word_vec_dim, 4 * hidden_dim)
        self.Wx /= np.sqrt(word_vec_dim)
        self.Wh = np.random.randn(hidden_dim, 4 * hidden_dim)
        self.Wh /= np.sqrt(word_vec_dim)
        self.b = np.zeros(4 * hidden_dim)

    def forward(self, x_sequence, h0):
        """Forward pass for a LSTM layer over an entire sequence of data. This assumes an input sequence composed of T
        vectors, each of dimension D. The LSTM uses a hidden size of H, and it works over a mini-batch containing N
        sequences.

        :param np.array x_sequence: Input data of shape (N, T, D)
        :param np.array h0: Initial hidden state of shape (N, H)
        :return np.array: Return hidden state over time of shape (N, T, H)
        """
        N, T, D = x_sequence.shape
        _, H = h0.shape

        # Create time series variables, i.e. hidden states over time and cell states over time
        hidden_states_over_t = np.zeros((N, T, H))
        cell_states_over_t = np.zeros((N, T, H))
        caches = dict()

        # Run the sequence
        prev_hidden_state = h0
        prev_cell_state = np.zeros(h0.shape)
        for t in range(T):
            hidden_state, cell_state, caches[t] = self.forward_step(x_sequence[:, t, :],
                                                                    prev_hidden_state,
                                                                    prev_cell_state)
            hidden_states_over_t[:, t, :] = hidden_state
            cell_states_over_t[:, t, :] = cell_state

        return hidden_states_over_t

    def forward_step(self, x, prev_hidden_state, prev_cell_state):
        """Forward pass for a single time step of the LSTM layer.

        :param np.array x: Input data of shape (N, D)
        :param np.array prev_hidden_state: Previous hidden state of shape (N, H)
        :param np.array prev_cell_state: Previous cell state of shape (N, H)
        """
        _, H = prev_hidden_state.shape

        # Compute activations
        acts = np.dot(x, self.Wx) + np.dot(prev_hidden_state, self.Wh) + self.b

        # Compute the internal gates
        input_gate = sigmoid(acts[:, 0:H])
        forget_gate = sigmoid(acts[:, H:2*H])
        output_gate = sigmoid(acts[:, 2*H:3*H])
        gain_gate = np.tanh(acts[:, 3*H:4*H])

        # Compute next states
        next_cell_state = forget_gate * prev_cell_state + input_gate * gain_gate
        next_hidden_state = output_gate * np.tanh(next_cell_state)

        # Cache the results
        cache = (next_hidden_state,
                 next_cell_state,
                 input_gate,
                 forget_gate,
                 output_gate,
                 gain_gate,
                 x,
                 prev_hidden_state,
                 prev_cell_state)

        return next_hidden_state, next_cell_state, cache


if __name__ == "__main__":
    """Glossary

    N: Mini-batch size
    T: Number of time steps
    H: Hidden dimension
    D: Word vector dimension
    """
    N, T, H, D = 10, 10, 20, 128
    layer = LSTMLayer(D, H)

    x_seq = np.random.rand(N, T, D)
    h0 = np.random.randn(N, H)
    print layer.forward(x_seq, h0).shape
