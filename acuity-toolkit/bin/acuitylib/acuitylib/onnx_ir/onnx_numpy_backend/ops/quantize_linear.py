from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def QuantizeLinear(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    y_scale = inputs[1]
    y_zero_point = np.array([0], dtype=np.uint8) if len(inputs) == 2 else inputs[2]
    min = 0
    max = 255
    if y_zero_point.dtype != np.uint8:
        min = -128
        max = 127
    y = np.round(x / y_scale) + y_zero_point
    y = np.clip(y, min, max)
    y = y.astype(y_zero_point.dtype)
    return y
