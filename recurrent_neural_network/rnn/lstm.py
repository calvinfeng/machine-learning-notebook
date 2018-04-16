import numpy as np
from helpers import sigmoid


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
        self.hidden_state_over_t = None
        self.cell_states_over_t = None
        self.caches = None

    def forward(self, input_sequence, h0, Wx=None, Wh=None, b=None):
        """Forward pass for a LSTM layer over an entire sequence of data. This assumes an input sequence composed of T
        vectors, each of dimension D. The LSTM uses a hidden size of H, and it works over a mini-batch containing N
        sequences.

        :param np.array input_sequence: Input data of shape (N, T, D)
        :param np.array h0: Initial hidden state of shape (N, H)
        :param np.array Wx: Optional input-to-hidden weight matrix, of shape (D, 4H)
        :param np.array Wh: Optional hidden-to-hidden weight matrix, of shape (H, 4H)
        :param np.array b: Optional bias vector, of shape (4H,)

        Returns np.array:
            Hidden state over time of shape (N, T, H)
        """
        if Wx is not None and Wh is not None and b is not None:
            self.Wx, self.Wh, self.b = Wx, Wh, b

        N, T, D = input_sequence.shape
        _, H = h0.shape

        # Cache the inputs and create time series variables, i.e. hidden states over time and cell states over time.
        self.input_sequence = input_sequence
        self.h0 = h0

        self.hidden_states_over_t = np.zeros((N, T, H))        
        self.cell_states_over_t = np.zeros((N, T, H))
        self.caches = dict()

        # Run the sequence
        prev_hidden_state = h0
        prev_cell_state = np.zeros(h0.shape)
        for t in range(T):
            hidden_state, cell_state, self.caches[t] = self._forward_step(input_sequence[:, t, :],
                                                                         prev_hidden_state,
                                                                         prev_cell_state)
            self.hidden_states_over_t[:, t, :] = hidden_state
            self.cell_states_over_t[:, t, :] = cell_state

            prev_hidden_state, prev_cell_state = hidden_state, cell_state

        return self.hidden_states_over_t

    def backward(self, grad_hidden_state_over_t):
        """Backward pass for a LSTM layer over an entire sequence of data.

        :param np.array grad_hidden_state: Upstream gradients of hidden states, of shape (N, T, H)

        Returns tuple:
            - grad_input_seq: Gradient of the input data, of shape (N, T, D)
            - grad_h0: Gradient of the initial hidden state, of shape (N, H)
            - grad_Wx: Gradient of input-to-hidden weight matrix, of shape (D, 4H)
            - grad_Wh: Gradient of hidden-to-hidden weight matrix, of shape (H, 4H)
            - grad_b: Gradient of biases, of shape (4H,)
        """
        N, T, H = grad_hidden_state_over_t.shape
        # grad_cell_state_over_t = np.zeros((N, T, H))

        grad_input_seq = np.zeros(self.input_sequence.shape)
        grad_Wx, grad_Wh, grad_b = np.zeros(self.Wx.shape), np.zeros(self.Wh.shape), np.zeros(self.b.shape)
        grad_prev_hidden_state = np.zeros((N, H))
        grad_prev_cell_state = np.zeros((N, H))

        for t in reversed(range(T)):
            time_step_result = self._backward_step(grad_hidden_state_over_t[:, t, :] + grad_prev_hidden_state,
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

        Returns tuple:
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
        cache = {
            'x': x,
            'next-c': next_hidden_state,
            'next-h': next_cell_state,
            'i-gate': input_gate,
            'f-gate': forget_gate,
            'o-gate': output_gate,
            'g-gate': gain_gate,
            'prev-h': prev_hidden_state,
            'prev-c': prev_cell_state
        }

        return next_hidden_state, next_cell_state, cache

    def _backward_step(self, grad_next_hidden_state, grad_next_cell_state, cache):
        """Backward pass for a single time step of the LSTM layer.

        :param np.array grad_next_hidden_state: Gradient of next hidden state, of shape (N, H)
        :param np.array grad_next_cell_state: Gradient of next cell state, of shape (N, H)
        :cache tuple cache: Cache object from the forward pass

        Returns tuple:
            - grad_x: Gradients of time step input, of shape (N, D)
            - grad_prev_hidden_state: Gradients of previous hidden state, of shape (N, H)
            - grad_prev_cell_state: Gradients of previous cell state, of shape (N, H)
            - grad_Wx: Gradients of input-to-hidden weights, of shape (D, 4H)
            - grad_Wh: Gradients of hidden-to-hidden weights, of shape (H, 4H)
            - grad_b: Gradients of bias, of shape (4H,)
        """
        # Note that grad_prev_c has two contributions, one from grad_next_cell_state and another one from grad_next_hidden_state
        grad_next_h_next_c = cache['o-gate'] * ( 1 - (np.tanh(cache['next-c']) * np.tanh(cache['next-c'])))
        
        grad_prev_cell_state = (grad_next_hidden_state * grad_next_h_next_c * cache['f-gate']) + (grad_next_cell_state + cache['f-gate'])
        
        # Each gate needs to go through the derivative of non-linearity
        grad_i_gate = (grad_next_hidden_state * grad_next_h_next_c * cache['g-gate']) + (grad_next_cell_state * cache['prev-c'])
        grad_i_gate = grad_i_gate * cache['i-gate'] * (1 - cache['i-gate'])

        grad_f_gate = (grad_next_hidden_state * grad_next_h_next_c * cache['prev-c']) + (grad_next_cell_state * cache['prev-c'])
        grad_f_gate = grad_f_gate * cache['f-gate'] * (1 - cache['f-gate'])

        grad_o_gate = grad_next_hidden_state * np.tanh(cache['next-c'])
        grad_o_gate = grad_o_gate * cache['o-gate'] * (1 - cache['o-gate'])

        grad_g_gate = (grad_next_hidden_state * grad_next_h_next_c * cache['i-gate']) + (grad_next_cell_state * cache['i-gate'])
        grad_g_gate = grad_g_gate * (1 - cache['g-gate'] * cache['g-gate'])

        # Now stack them
        grad_act = np.concatenate((grad_i_gate, grad_f_gate, grad_o_gate, grad_g_gate), axis=1)
        
        # And then do the same ol' gradient calculations
        grad_x = np.dot(grad_act, self.Wx.T)
        grad_prev_hidden_state = np.dot(grad_act, self.Wh.T)
        grad_Wx = np.dot(cache['x'].T, grad_act)
        grad_Wh = np.dot(cache['prev-h'].T, grad_act)
        grad_b = np.sum(grad_act, axis=0)

        return grad_x, grad_prev_hidden_state, grad_prev_cell_state, grad_Wx, grad_Wh, grad_b 
