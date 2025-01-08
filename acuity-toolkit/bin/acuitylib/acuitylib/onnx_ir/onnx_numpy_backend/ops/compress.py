from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Compress(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    condition = inputs[1]
    data = inputs[0]
    return np.compress(condition, data, axis=attr.get('axis', None))
