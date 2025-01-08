from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def SpaceToDepth(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    b,c,h,w = inputs[0].shape
    blocksize = attr['blocksize']
    res = np.reshape(inputs[0], [b, c , h // blocksize, blocksize, w // blocksize, blocksize])
    res = np.transpose(res, [0, 1, 2, 4, 3, 5])
    res = np.reshape(res, [b, c *(blocksize**2), h//blocksize, w//blocksize])
    return res

