from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import copy

def SequenceInsert(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    seq = copy.copy(inputs[0])
    t = inputs[1]
    if len(inputs) == 2:
        seq.append(t)
        return seq
    else:
        pos = int(inputs[2])
        seq.insert(pos, t)
        return seq
