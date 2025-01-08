from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def CumSum(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    axis = None if len(inputs) < 2 else int(inputs[1])
    if x.ndim == 1:
        exclusive = attr.get('exclusive',0) == 1
        reverse = attr.get('reverse', 0) == 1
        if reverse:
            x = x[::-1]
        res = np.cumsum(x, axis)
        if exclusive :
            t = np.zeros_like(res)
            t[1:] = res[0:-1]
            res = t
        if reverse:
            res = res[::-1]
    else:
        res = np.cumsum(x, axis)
    return res
