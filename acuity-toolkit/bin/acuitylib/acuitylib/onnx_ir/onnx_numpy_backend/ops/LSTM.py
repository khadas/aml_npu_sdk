from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def _convert_weight(weight, direction = 'forward'):
    # onnx weight:  iofc
    # torch weight: ifco
    def _format_weight(weight):
        weight = np.squeeze(weight, axis=0)
        w = np.split(weight, 4, axis=0)
        new_w = np.concatenate([w[0], w[2], w[3], w[1]], axis=0)
        return new_w

    if direction == 'bidirectional':
        weight_f, weight_r = np.split(weight, 2, axis=0)
        new_w_f = _format_weight(weight_f)
        new_w_r = _format_weight(weight_r)
    else:
        new_w_f = _format_weight(weight)
        new_w_r = None
    return new_w_f, new_w_r

def _convert_bias(bias, direction = 'forward'):
    def _format_bias(bias):
        b = np.squeeze(bias, axis=0)
        ih, hh = np.split(b, 2, axis=0)
        ih = np.split(ih, 4, axis=0)
        hh = np.split(hh, 4, axis=0)
        new_ih = np.concatenate([ih[0], ih[2], ih[3], ih[1]], axis=0)
        new_hh = np.concatenate([hh[0], hh[2], hh[3], hh[1]], axis=0)
        return new_ih, new_hh

    if direction == 'bidirectional':
        bias_f, bias_r = np.split(bias, 2, axis=0)
        new_f_ih, new_f_hh = _format_bias(bias_f)
        new_r_ih, new_r_hh = _format_bias(bias_r)
    else:
        new_f_ih, new_f_hh = _format_bias(bias)
        new_r_ih = new_r_hh = None

    return new_f_ih, new_f_hh, new_r_ih, new_r_hh

def LSTM(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    X, W, R = inputs[0:3]
    timestep, batch_size, input_size = X.shape
    num_directions = W.shape[0]
    hidden_size = attr.get('hidden_size')
    direction = attr.get('direction', 'forward')
    activations = attr.get('activations', None)

    if len(inputs) > 3 and isinstance(inputs[3], np.ndarray) and len(inputs[3].shape) > 0:
        B = inputs[3]
    else:
        B = np.zeros([num_directions, 8*hidden_size], X.dtype)
    if len(inputs) > 4 and isinstance(inputs[4], np.ndarray) and len(inputs[4].shape) > 0:
        sequence_lens = inputs[4]
    else:
        sequence_lens = X.shape[1]
    if len(inputs) > 5 and isinstance(inputs[5], np.ndarray) and len(inputs[5].shape) > 0:
        initial_h = inputs[5]
    else:
        initial_h = np.zeros([num_directions, batch_size, hidden_size], X.dtype)
    if len(inputs) > 6 and isinstance(inputs[6], np.ndarray) and len(inputs[6].shape) > 0:
        initial_c = inputs[6]
    else:
        initial_c = np.zeros([num_directions, batch_size, hidden_size], X.dtype)
    if len(inputs) > 7 and isinstance(inputs[7], np.ndarray) and len(inputs[7].shape) > 0:
        P = inputs[7]
    else:
        P = np.zeros([num_directions, 3*hidden_size], X.dtype)

    if direction == 'bidirectional':
        bidirectional = True
    else:
        bidirectional = False

    # TODO: The PyTroch LSTM don't have peephole and parameter 'activations', so the this numpy_backend can't
    #       hanndle them.
    import torch
    lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)

    weight_f_ih, weight_r_ih = _convert_weight(W, direction=direction)
    weight_f_hh, weight_r_hh = _convert_weight(R, direction=direction)
    bias_f_ih, bias_f_hh, bias_r_ih, bias_r_hh = _convert_bias(B, direction=direction)

    lstm.weight_ih_l0 = torch.nn.Parameter(torch.Tensor(weight_f_ih))
    lstm.weight_hh_l0 = torch.nn.Parameter(torch.Tensor(weight_f_hh))
    lstm.bias_ih_l0 = torch.nn.Parameter(torch.Tensor(bias_f_ih))
    lstm.bias_hh_l0 = torch.nn.Parameter(torch.Tensor(bias_f_hh))
    if bidirectional is True:
        lstm.weight_ih_l0_reverse = torch.nn.Parameter(torch.Tensor(weight_r_ih))
        lstm.weight_hh_l0_reverse = torch.nn.Parameter(torch.Tensor(weight_r_hh))
        lstm.bias_ih_l0_reverse = torch.nn.Parameter(torch.Tensor(bias_r_ih))
        lstm.bias_hh_l0_reverse = torch.nn.Parameter(torch.Tensor(bias_r_hh))

    input = torch.Tensor(X)
    h0 = torch.Tensor(initial_h)
    c0 = torch.Tensor(initial_c)

    output, (hn, cn) = lstm(input, (h0, c0))
    res = output.detach().numpy()
    hn = hn.detach().numpy()
    cn = cn.detach().numpy()

    # onnx res:    [timestep, num_directions, batch_size, hidden_size]
    # pytorch res: [timestep, batch, num_directions * hidden_size]
    res = np.reshape(res, [timestep, batch_size, num_directions, hidden_size])
    res = np.transpose(res, [0, 2, 1, 3])
    return [res, hn, cn]
