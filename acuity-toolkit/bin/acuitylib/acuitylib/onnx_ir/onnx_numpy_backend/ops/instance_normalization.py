from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

def InstanceNormalization(inputs, outputs, attr=None, op_version=11):
    # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    scale = inputs[1]
    B = inputs[2]
    epsilon = attr.get('epsilon', 1e-05)

    return _instancenorm_test_mode(x, scale, B, epsilon)

