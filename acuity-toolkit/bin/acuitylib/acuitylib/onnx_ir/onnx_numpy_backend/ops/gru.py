from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

class gru_helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # GRU Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        LBR = str('linear_before_reset')
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr

        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray, list, dict, int) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray, list, dict, int) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        h_list = []
        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(np.dot(x, np.transpose(w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh)
            h_linear = self.g(np.dot(x, np.transpose(w_h)) + r * (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh)
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]

def GRU(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    X, W, R = inputs[0:3]
    seq_length, batch_size, input_size = X.shape
    num_directions, _, _ = W.shape
    activation_alpha = attr.get('activation_alpha', 0.01)
    activation_beta = attr.get('activation_beta', None)
    activations = attr.get('activations', None)
    clip = attr.get('clip', None)
    direction = attr.get('direction', 'forward')
    hidden_size = attr.get('hidden_size')
    linear_before_reset = attr.get('linear_before_reset', 0)

    B = \
        np.zeros([num_directions, 6*hidden_size], X.dtype) \
        if len(inputs) < 4 else \
        inputs[3]
    sequence_lens = inputs[4] if len(inputs) >= 5 else X.shape[1]
    initial_h = \
        np.zeros([num_directions, batch_size, hidden_size], X.dtype) \
        if len(inputs) < 6 else\
        inputs[5]

    gru = gru_helper(X=X,
                     W=W,
                     R=R,
                     B=B,
                     sequence_lens=sequence_lens,
                     initial_h=initial_h,
                     linear_before_reset=linear_before_reset)
    output, Y_h = gru.step()
    if len(outputs) == 1:
        return output
    else:
        return [output.astype(X.dtype), Y_h.astype(X.dtype)]
