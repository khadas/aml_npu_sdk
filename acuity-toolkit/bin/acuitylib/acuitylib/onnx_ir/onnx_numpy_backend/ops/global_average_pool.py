from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def GlobalAveragePool(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray

    spatial_ndim = inputs[0].ndim - 2
    res = np.average(inputs[0], axis=tuple(range(2, 2+spatial_ndim)))
    for _ in range(spatial_ndim):
        res = np.expand_dims(res, -1)
    return res