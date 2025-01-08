from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ConvInteger(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x, w = inputs[0:2]
    if len(inputs) <= 2 or (isinstance(inputs[2], str) and inputs[2] == ''):
        x_zero_point = np.array([0], dtype=x.dtype)
    else:
        x_zero_point = inputs[2]

    if len(inputs) <= 3 or (isinstance(inputs[3], str) and inputs[3] == ''):
        w_zero_point = np.array([0], dtype=w.dtype)
    else:
        w_zero_point = inputs[3]

    x = x.astype(np.int)
    w = w.astype(np.int)
    x_zero_point = x_zero_point.astype(np.int)
    w_zero_point = w_zero_point.astype(np.int)
    x = x - x_zero_point
    w = w - w_zero_point
    x = x.astype(np.float32)
    w = w.astype(np.float32)

    from .conv import Conv
    conv_inputs = [x, w]
    conv_res = Conv(conv_inputs, ['Conv_Out'], attr, op_version)
    conv_res = conv_res.astype(np.int32)
    return conv_res
