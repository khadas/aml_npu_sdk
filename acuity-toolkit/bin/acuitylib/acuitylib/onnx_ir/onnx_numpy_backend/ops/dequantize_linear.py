from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def DequantizeLinear(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x, x_scale= inputs[0], inputs[1]
    x_zero_point = np.array([0], dtype=inputs[0].dtype) if len(inputs) == 2 else inputs[2]
    x = x.astype(np.int)
    x_zero_point = x_zero_point.astype(np.int)
    res = (x - x_zero_point) * x_scale
    res = res.astype(np.float32)
    return res
