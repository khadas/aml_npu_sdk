from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def LpNormalization(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    axis = -1 # onnx default value
    p = 2 # onnx default value
    if 'axis' in attr:
        axis = attr['axis']
    if 'p' in attr:
        p = attr['p']
    return np.linalg.norm(x=x, ord=p, axis=axis)

