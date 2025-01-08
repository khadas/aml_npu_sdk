from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Flatten(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    shape = x.shape
    axis = attr.get('axis',1)
    if axis != 0:
        new_shape = [np.prod(shape[0:axis]).astype(int), -1]
    else :
        new_shape = [1] + [-1]
    return np.reshape(x, new_shape)
