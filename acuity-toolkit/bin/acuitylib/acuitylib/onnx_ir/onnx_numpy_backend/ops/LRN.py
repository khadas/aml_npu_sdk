from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import math

def LRN(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    C = x.shape[1]
    square_sum = np.zeros_like(x).astype(np.float32)
    alpha = attr.get('alpha', 0.0001)
    beta = attr.get('beta', 0.75)
    bias = attr.get('bias', 1.0)
    nsize = attr['size']
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(x[n,
                                     max(0, c - int(math.floor((nsize - 1) / 2))):min(C, c + int(
                                         math.ceil((nsize - 1) / 2)) + 1),
                                     h,
                                     w] ** 2)
    y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
    return y
