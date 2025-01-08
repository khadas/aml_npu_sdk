from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Celu(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    alpha = attr.get('alpha', 1.0)
    result = np.maximum(inputs[0], float(0)) + np.minimum(float(0), alpha * (np.exp(inputs[0] / alpha) - 1))
    return result
