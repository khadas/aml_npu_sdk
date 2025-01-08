from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def GatherElements(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    def gather_elements(data, indices, axis=0):  # type: ignore
        data_swaped = np.swapaxes(data, 0, axis)
        index_swaped = np.swapaxes(indices, 0, axis)
        gathered = np.choose(index_swaped, data_swaped, mode='wrap')
        y = np.swapaxes(gathered, 0, axis)
        return y

    data = inputs[0]
    indices = inputs[1]
    axis = attr.get('axis', 0)
    return gather_elements(data, indices, axis)

