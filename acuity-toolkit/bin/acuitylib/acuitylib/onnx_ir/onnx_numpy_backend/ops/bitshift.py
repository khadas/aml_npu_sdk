from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def BitShift(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    if attr['direction'] == 'LEFT':
        return inputs[0] << inputs[1]
    else:
        return inputs[0] >> inputs[1]
