from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def QLinearConv(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point = inputs[0:8]
    from .dequantize_linear import DequantizeLinear
    x_f = DequantizeLinear([x, x_scale, x_zero_point], ['out_x'])
    w_f = DequantizeLinear([w, w_scale, w_zero_point], ['out_w'])
    from .conv import Conv
    conv_inputs = [x_f, w_f]
    if len(inputs) > 8:
        #conv_inputs.append(inputs[-1])
        b = inputs[8]
        b_f = DequantizeLinear([b, x_scale * w_scale, np.array([0]*w.shape[0], dtype=inputs[0].dtype)], ['out_b'])
        conv_inputs.append(b_f)
    y_f = Conv(conv_inputs, ['Conv_Out'], attr, op_version)
    from .quantize_linear import QuantizeLinear
    y_q = QuantizeLinear([y_f, y_scale, y_zero_point], ['out_y_q'])
    return y_q
