from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Squeeze(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    axis = None
    if op_version < 13:
        if attr is not None and 'axes' in attr.keys():
            axis = tuple(attr.get('axes'))
    else:  # axes is moved to inputs
        if len(inputs) == 2 and inputs[1].ndim == 1:
            axis = tuple(inputs[1].tolist())

    return np.squeeze(inputs[0], axis=axis)
