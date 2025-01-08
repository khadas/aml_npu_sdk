from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Clip(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    if op_version <= 6:
        min = attr.get('min', -np.inf)
        max = attr.get('max', np.inf)
    else:
        in_len = len(inputs)

        min = -np.inf if in_len < 2 or (isinstance(inputs[1], str) and inputs[1] == '') else inputs[1]
        max = np.inf if in_len < 3 or (isinstance(inputs[2], str) and inputs[2] == '') else inputs[2]

    return np.clip(inputs[0], min, max)
