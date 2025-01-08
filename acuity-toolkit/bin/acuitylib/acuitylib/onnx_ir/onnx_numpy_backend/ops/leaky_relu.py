from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def LeakyRelu(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    alpha = attr.get('alpha', 0.01)
    return np.clip(inputs[0], 0, np.inf) + np.clip(inputs[0], -np.inf, 0) * alpha
