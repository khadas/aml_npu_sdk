from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Concat(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    real_inputs = [i for i in inputs if not (isinstance(i, str) == True and i == '')]
    if len(real_inputs) == 1:
        return real_inputs[0]
    return np.concatenate(real_inputs, attr['axis'])
