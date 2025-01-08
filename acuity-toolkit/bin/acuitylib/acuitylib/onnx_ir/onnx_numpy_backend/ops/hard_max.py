from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def hardmax_2d(x):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    return np.eye(x.shape[1], dtype=x.dtype)[np.argmax(x, axis=1)]

def Hardmax(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    shape = x.shape
    axis = attr.get('axis', 1)
    if axis < 0:
        axis = x.ndim + axis
    s_base = max(1, np.prod(shape[0:axis]))
    s_left = np.prod(shape[axis:])
    res = hardmax_2d(np.reshape(x, [s_base, s_left])).reshape(shape)
    return res
