from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def SplitToSequence(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    axis = attr.get('axis', 0)
    keepdims = attr.get('keepdims', 1)
    if len(inputs) == 1 or isinstance(inputs[1], str):
        sections = x.shape[axis]
    else:
        sections = inputs[1]

    if keepdims == 0:
        return [np.squeeze(r, axis=axis) for r in  np.split(inputs[0], indices_or_sections=sections, axis=axis)]
    else:
        return np.split(inputs[0], indices_or_sections=sections, axis=axis)


