from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def reshape_reference_implementation(data, shape):  # type: (np.ndarray, np.ndarray) -> np.ndarray
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0
    new_shape = np.copy(shape).astype(np.int32)
    zeros_index = np.where(shape == 0)
    new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped

def Reshape(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    data = inputs[0]
    if op_version < 5:
        shape = attr['shape']
    else:
        shape = inputs[1]
    return reshape_reference_implementation(data, shape)
