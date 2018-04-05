import numpy as np
from numpy import dot, tanh


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class LSTMModel(object):
    """Long short-term memory recurrent neural network

    :param integer hidden_dim: Size of the hidden layer of neurons
    :param integer input_dim: Please note that input dimension is the same as output dimension for this character model
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim, self.output_dim = input_dim, input_dim
        self.hidden_dim = hidden_dim
        self.params = {
            'Wi': np.random.randn(2 * self.hidden_dim, self.input_dim) * 0.01,
            'Wf': np.random.randn(2 * self.hidden_dim, self.input_dim) * 0.01,
            'Wo': np.random.randn(2 * self.hidden_dim, self.input_dim) * 0.01,
            'Wg': np.random.randn(2 * self.hidden_dim, self.input_dim) * 0.01,
            'Why': np.random.randn(self.output_dim, self.hidden_dim) * 0.01,
            'Bi': np.random.zeros((2 * self.hidden_dim, 1)),
            'Bf': np.random.zeros((2 * self.hidden_dim, 1)),
            'Bo': np.random.zeros((2 * self.hidden_dim, 1)),
            'Bg': np.random.zeros((2 * self.hidden_dim, 1)),
            'By': np.zeros((self.output_dim, 1))
        }

    def loss(self, input_seq, target_seq, prev_hidden_state, prev_cell_state):
        x_states, hidden_states, cell_states, y_states, prob_states = {}, {}, {}, {}
        hidden_states[-1] = np.copy(prev_hidden_state)
        cell_states[-1] = np.copy(prev_cell_state)

        loss = 0
        for t in xrange(len(input_seq)):
            # Encode input state in 1-of-k representation
            x_states[t] = np.zeros((self.input_dim, 1))
            x_states[t][input_seq[t]] = 1

            # Compute cell states
            i_gate = sigmoid(dot(self.params['Wi'], np.concatenate(hidden_state[t-1], x_states[t])) + self.params['Bi'])
            f_gate = sigmoid(dot(self.params['Wf'], np.concatenate(hidden_state[t-1], x_states[t])) + self.params['Bf'])
            o_gate = sigmoid(dot(self.params['Wo'], np.concatenate(hidden_state[t-1], x_states[t])) + self.params['Bo'])
            g_gate = tanh(dot(self.params['Wf'], np.concatenate(hidden_state[t-1], x_states[t])) + self.params['Bg'])

            cell_states[t] = f_gate * cell_states[t-1] + i_gate * g_gate
            hidden_states[t] = o_gate * tanh(cell_states[t])

            # Compute output state a.k.a. unnomralized log probability using current hidden state
            y_states[t] = dot(self.params['Why'], hidden_states[t]) + self.params['By']

            # Compute softmax probability state using the output state
            prob_states[t] = exp(y_states[t]) / np.sum(exp(y_states[t]))

            loss += -np.log(prob_states[t][target_seq[t], 0])  # Remember that prob is an (O, 1) vector.
