from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Range(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    start = inputs[0]
    limit = inputs[1]
    delta = inputs[2]
    return np.arange(start, limit, delta, dtype=delta.dtype)
