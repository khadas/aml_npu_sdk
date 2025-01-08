from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def IsInf(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    detect_positive = attr.get('detect_positive', 1)
    detect_negative = attr.get('detect_negative', 1)
    if detect_positive==0:
        return np.isneginf(x)
    if detect_negative == 0:
        return np.isposinf(x)
    return np.isinf(x)
