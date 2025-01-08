from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def one_hot(indices, depth, axis=-1, dtype=np.float32):  # type: ignore
    ''' Compute one hot from indices at a specific axis '''
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += (rank + 1)
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
    values = np.reshape(np.mod(values, depth), ls + (1,) + rs)
    return np.asarray(targets == values, dtype=dtype)


def OneHot(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    indices = inputs[0]
    depth = inputs[1]
    value = inputs[2]
    off_value, on_value = value
    axis = attr.get('axis', -1)
    y = one_hot(indices, depth, axis=axis, dtype=on_value.dtype)
    y = y * (on_value - off_value) + off_value
    return y
