from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ArgMax(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    axis = attr.get('axis', 0)
    keep_dims = attr.get('keepdims', 1)
    result = np.argmax(inputs[0], axis=axis)
    if keep_dims == 1:
        result = np.expand_dims(result, axis)
    return result

