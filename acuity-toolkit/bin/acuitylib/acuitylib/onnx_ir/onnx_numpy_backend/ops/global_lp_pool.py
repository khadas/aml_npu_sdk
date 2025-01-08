from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def GlobalLpPool(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    spatial_shape = inputs[0].ndim - 2
    res = np.linalg.norm(inputs[0], axis=tuple(range(spatial_shape, spatial_shape+2)), ord=attr.get('p', 2))
    for _ in range(spatial_shape):
        res = np.expand_dims(res, -1)
    return res