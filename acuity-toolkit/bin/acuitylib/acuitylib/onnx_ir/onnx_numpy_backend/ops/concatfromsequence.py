from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ConcatFromSequence(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    new_axis = attr.get('new_axis',0) != 0
    if new_axis:
        res = np.stack(inputs[0], attr['axis'])
    else:
        res = np.concatenate(inputs[0], attr['axis'])
    return res
