from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def HardSigmoid(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    alpha = attr.get('alpha', 0.2)
    beta = attr.get('beta', 0.5)
    return np.clip(inputs[0]*alpha + beta, 0, 1)
