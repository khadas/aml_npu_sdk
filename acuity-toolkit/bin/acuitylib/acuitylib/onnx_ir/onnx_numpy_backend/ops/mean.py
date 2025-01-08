from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Mean(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    sum = inputs[0]
    if len(inputs) > 1:
        for data in inputs[1:]:
            sum = data + sum
    mean = sum / len(inputs)
    return mean
