from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ReduceMax(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    return np.maximum.reduce(inputs[0], axis=tuple(attr['axes']) if 'axes' in attr else None, keepdims=attr.get('keepdims', 1)==1)

