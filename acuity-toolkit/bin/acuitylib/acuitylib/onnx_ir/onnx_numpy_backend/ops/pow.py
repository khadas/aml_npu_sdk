from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Pow(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    return np.power(inputs[0], inputs[1])
