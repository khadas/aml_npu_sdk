from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

class rnn_helper():
    def __init__(self, **params):  # type: (**Any) -> None
        # RNN Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[str(W)].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(2 * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray, list, dict, int) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        h_list = []
        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            H = self.f(np.dot(x, np.transpose(self.W)) + np.dot(H_t, np.transpose(self.R)) + np.add(
                *np.split(self.B, 2)))
            h_list.append(H)
            H_t = H
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]

def RNN(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    X, W, R = inputs[0:3]
    seq_length, batch_size, input_size = X.shape
    num_directions, _, _ = W.shape
    activation_alpha = attr.get('activation_alpha', 0.01)
    activation_beta = attr.get('activation_beta', None)
    activations = attr.get('activations', ['Tanh', 'Tanh'])
    clip = attr.get('clip', None)
    direction = attr.get('direction', 'forward')
    hidden_size = attr.get('hidden_size')

    B = \
        np.zeros([num_directions, 2*hidden_size], X.dtype) \
        if len(inputs) < 4 else \
        inputs[3]
    sequence_lens = inputs[4] if len(inputs) >= 5 else X.shape[1]
    initial_h = \
        np.zeros([num_directions, batch_size, hidden_size], X.dtype) \
        if len(inputs) < 6 else\
        inputs[5]

    rnn = rnn_helper(X=X,
                     W=W,
                     R=R,
                     B=B,
                     sequence_lens=sequence_lens,
                     initial_h=initial_h)
    Y, Y_h = rnn.step()
    if len(outputs) == 1:
        return Y
    else:
        return [Y.astype(X.dtype), Y_h.astype(X.dtype)]
