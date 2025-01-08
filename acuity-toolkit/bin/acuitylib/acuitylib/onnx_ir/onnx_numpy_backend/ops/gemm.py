from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Gemm(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    alpha = attr.get('alpha', 1.0)
    beta = attr.get('beta', 1.0)
    transA = attr.get('transA', 0) == 1
    transB = attr.get('transB', 0) == 1
    from .flatten import Flatten
    A = inputs[0]
    A = Flatten([A], ['one'], dict())
    if transA:
        A = A.T
    B = inputs[1]
    if transB:
        B = B.T
    C = np.array(0) if len(inputs) < 3 else inputs[2]
    res = alpha * np.dot(A, B) + beta * C
    return res