from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def MatMulInteger(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    A, B = inputs[0:2]
    if len(inputs) <= 2 or (isinstance(inputs[2], str) and inputs[2] == ''):
        a_zero_point = np.array([0], dtype=A.dtype)
    else:
        a_zero_point = inputs[2]

    if len(inputs) <= 3 or (isinstance(inputs[3], str) and inputs[3] == ''):
        b_zero_point = np.array([0], dtype=B.dtype)
    else:
        b_zero_point = inputs[3]

    A = A.astype(np.int)
    B = B.astype(np.int)
    a_zero_point = a_zero_point.astype(np.int)
    b_zero_point = b_zero_point.astype(np.int)
    Y = np.matmul(A-a_zero_point, B-b_zero_point)
    Y = Y.astype(np.int32)
    return Y
