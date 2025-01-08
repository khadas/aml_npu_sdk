from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Shrink(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    lambd = attr.get('lambd', 0.5)
    bias = attr.get('bias', 0.0)
    y1 = np.zeros_like(x)
    y2 = np.zeros_like(x)
    y3 = np.zeros_like(x)

    y1[x<-lambd] = x[x<-lambd] + bias
    # y2[np.abs(x)<=lambd] = x[np.abs(x)<=lambd]
    y3[x>lambd] = x[x>lambd] - bias
    res = y1 + y2 + y3
    return res
