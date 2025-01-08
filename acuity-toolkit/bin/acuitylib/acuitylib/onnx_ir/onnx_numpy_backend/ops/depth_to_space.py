from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def DepthToSpace(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    b,c,h,w = inputs[0].shape
    blocksize = attr['blocksize']
    if attr.get('mode', 'DCR') == 'DCR':
      res = np.reshape(inputs[0], [b, blocksize, blocksize, c // (blocksize ** 2), h, w])
      res = np.transpose(res, [0, 3, 4, 1, 5, 2])
    else:
      res = np.reshape(inputs[0], [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
      res = np.transpose(res, [0, 1, 4, 2, 5, 3])
    res = np.reshape(res, [b, c //(blocksize**2), h*blocksize, w*blocksize])
    return res
