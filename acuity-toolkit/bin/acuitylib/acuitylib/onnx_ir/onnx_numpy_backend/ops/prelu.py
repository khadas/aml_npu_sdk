from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from acuitylib.acuitylog import AcuityLog as al

def PRelu(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    slope = inputs[1]
    if slope.ndim == 1 and slope.shape[0] != 1:
        # slope can unidirectional broadcastable to x
        ss = slope.shape[0]
        source_shape = x.shape
        shape_index = source_shape.index(ss)
        if shape_index == -1:
            al.e("The Shape can't boardcast {} and {}".format('x'.join([str(s) for s in source_shape]), str(ss)))
        shape = list()
        for i in range(x.ndim):
            if i != shape_index:
                shape.append(1)
            else:
                shape.append(ss)
        slope = np.reshape(slope, shape)

    return np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

