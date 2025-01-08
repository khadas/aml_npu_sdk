from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def QLinearMatMul(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point = inputs
    from .dequantize_linear import DequantizeLinear
    a_f = DequantizeLinear([a, a_scale, a_zero_point], ['out_a'])
    b_f = DequantizeLinear([b, b_scale, b_zero_point], ['out_b'])
    from .mat_mul import MatMul
    y_f = MatMul([a_f, b_f], ['out_y'])
    from .quantize_linear import QuantizeLinear
    y_q = QuantizeLinear([y_f, y_scale, y_zero_point], ['out_y_q'])
    return y_q
