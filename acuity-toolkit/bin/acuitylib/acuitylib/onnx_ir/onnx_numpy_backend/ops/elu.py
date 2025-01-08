from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Elu(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    return np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * attr.get('alpha', 1.0)
