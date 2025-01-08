from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore


def Einsum(inputs, outputs, attr=None, op_version=12):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    subscripts = attr.get('equation')
    result = np.einsum(subscripts, *inputs)
    return result
