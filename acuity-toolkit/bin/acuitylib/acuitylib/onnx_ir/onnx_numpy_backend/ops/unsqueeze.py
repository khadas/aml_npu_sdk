from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Unsqueeze(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    res = inputs[0]
    axes = []
    if op_version < 13:
        if attr is not None and 'axes' in attr.keys():
            axes = attr.get('axes')
    else:  # axes is moved to inputs
        if len(inputs) == 2 and inputs[1].ndim == 1:
            axes = inputs[1].tolist()

    for axis in sorted(axes):
      res = np.expand_dims(res, axis=axis)
    return res
