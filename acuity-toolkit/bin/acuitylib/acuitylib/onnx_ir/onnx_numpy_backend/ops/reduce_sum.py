from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ReduceSum(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    keep_dims = attr.get('keepdims', 1) == 1
    axes = None
    if op_version < 13:
        if attr is not None and 'axes' in attr.keys():
            axes = tuple(attr['axes'])
    else:
        if len(inputs) == 1 or (len(inputs) == 2 and np.size(inputs[1]) == 0):
            if 'noop_with_empty_axes' in attr and attr.get('noop_with_empty_axes', 0) == 1:
                return inputs[0]
        elif len(inputs) == 2 and inputs[1].ndim == 1:
            axes = tuple(inputs[1].tolist())

    return np.sum(inputs[0], keepdims=keep_dims, axis=axes)



