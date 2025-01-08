from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ThresholdedRelu(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    alpha = attr.get('alpha', 1.0)
    y = np.clip(x, alpha, np.inf)  # expected output [0., 0., 0., 0., 2.2]
    y[y == alpha] = 0
    return y
