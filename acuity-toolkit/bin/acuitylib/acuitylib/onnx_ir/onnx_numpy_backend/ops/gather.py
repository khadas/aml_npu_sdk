from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Gather(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    indices = inputs[1].astype(np.int32)
    return np.take(inputs[0], indices, axis=attr.get('axis', 0))
