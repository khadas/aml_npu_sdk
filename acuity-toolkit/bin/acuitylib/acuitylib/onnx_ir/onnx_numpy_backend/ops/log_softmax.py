from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def logsoftmax_2d(x):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

def LogSoftmax(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    shape = inputs[0].shape
    axis = attr.get('axis', 1)
    s_base = max(1, np.prod(shape[0:axis]))
    s_left = np.prod(shape[axis:])
    res = logsoftmax_2d(np.reshape(x, [s_base, s_left])).reshape(shape)
    return res