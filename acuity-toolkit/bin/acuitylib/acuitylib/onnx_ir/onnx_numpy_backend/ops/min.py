from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Min(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    res = inputs[0]
    if len(inputs) > 1:
        for data in inputs[1:]:
            res = np.minimum(data, res)
    return res