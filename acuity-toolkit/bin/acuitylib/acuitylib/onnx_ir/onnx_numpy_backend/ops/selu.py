from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Selu(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    alpha = attr.get('alpha', 1.67326319217681884765625)
    gamma = attr.get('gamma', 1.05070102214813232421875)
    return np.clip(x, 0, np.inf) * gamma + \
    (np.exp(np.clip(x, -np.inf, 0)) - 1) * alpha * gamma
