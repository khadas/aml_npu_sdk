from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Mod(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    y = inputs[1]
    fmod = attr.get('fmod', 0) == 1
    if fmod:
        return np.fmod(x, y)
    else:
        return np.mod(x, y)
