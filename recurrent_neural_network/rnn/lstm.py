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

        # Required states for back-propagation
        self.h0 = None
        self.input_sequence = None
        self.cell_states_over_t = None
        self.caches = None

    def forward(self, input_sequence, h0):
        """Forward pass for a LSTM layer over an entire sequence of data. This assumes an input sequence composed of T
        vectors, each of dimension D. The LSTM uses a hidden size of H, and it works over a mini-batch containing N
        sequences.

        :param np.array input_sequence: Input data of shape (N, T, D)
        :param np.array h0: Initial hidden state of shape (N, H)
        :return np.array: Return hidden state over time of shape (N, T, H)
        """
        N, T, D = input_sequence.shape
        _, H = h0.shape

        # Cache the inputs and create time series variables, i.e. hidden states over time and cell states over time.
        self.h0 = h0
        self.input_sequence = input_sequence
        self.cell_states_over_t = np.zeros((N, T, H))
        self.caches = dict()
        hidden_states_over_t = np.zeros((N, T, H))

        # Run the sequence
        prev_hidden_state = h0
        prev_cell_state = np.zeros(h0.shape)
        for t in range(T):
            hidden_state, cell_state, self.caches[t] = self._forward_step(input_sequence[:, t, :],
                                                                         prev_hidden_state,
                                                                         prev_cell_state)
            hidden_states_over_t[:, t, :] = hidden_state
            self.cell_states_over_t[:, t, :] = cell_state

        return hidden_states_over_t

    def backward(self, grad_hidden_state_over_t):
        """Backward pass for a LSTM layer over an entire sequence of data.

        :param np.array grad_hidden_state: Upstream gradients of hidden states, of shape (N, T, H)

        Returns a tuple of:
            - grad_input_seq: Gradient of the input data, of shape (N, T, D)
            - grad_h0: Gradient of the initial hidden state, of shape (N, H)
            - grad_Wx: Gradient of input-to-hidden weight matrix, of shape (D, 4H)
            - grad_Wh: Gradient of hidden-to-hidden weight matrix, of shape (H, 4H)
            - grad_b: Gradient of biases, of shape (4H,)
        """
        N, T, H = grad_hidden_state_over_t.shape
        grad_cell_state_over_t = np.zeros((N, T, H))

        grad_input_seq = np.zeros(self.input_sequence.shape)
        grad_Wx, grad_Wh, grad_b = np.zeros(self.Wx.shape), np.zeros(self.Wh.shape), np.zeros(self.b.shape)
        grad_prev_hidden_state = np.zeros((N, H))
        grad_prev_cell_state = np.zeros((N, H))

        for t in reversed(range(T)):
            time_step_result = self.backward_step(grad_hidden_state_over_t[:, t, :] + grad_prev_hidden_state,
                                                  grad_prev_cell_state,
                                                  self.caches[t])
            grad_input_seq[:, t, :] = time_step_result[0]
            grad_prev_hidden_state = time_step_result[1]
            grad_prev_cell_state = time_step_result[2]

            # Accumulate
            grad_Wx += time_step_result[3]
            grad_Wh += time_step_result[4]
            grad_b += time_step_result[5]

        # Gradient of the initial hidden state is the last grad_prev_hidden_state
        grad_h0 = grad_prev_hidden_state

        return grad_input_seq, grad_h0, grad_Wx, grad_Wh, grad_b

    def _forward_step(self, x, prev_hidden_state, prev_cell_state):
        """Forward pass for a single time step of the LSTM layer.

        :param np.array x: Input data of shape (N, D)
        :param np.array prev_hidden_state: Previous hidden state of shape (N, H)
        :param np.array prev_cell_state: Previous cell state of shape (N, H)

        Returns a tuple of:
            - next_hidden_state: Next hidden state, of shape (N, H)
            - next_cell_state: Next cell state, of shape (N, H)
            - cache: Tuple of values needed for back-propagation
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

    def _backward_step(self):
        pass

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
