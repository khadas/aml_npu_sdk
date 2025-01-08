from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Pad(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    if op_version < 11:
        pads = attr['pads']
        const_value = attr.get('value', 0.0)
    else:
        pads = inputs[1]
        const_value = 0.0 if len(inputs) < 3 else inputs[2]
    pads = np.reshape(pads, [2, -1])
    pads = np.transpose(pads)
    pad_mode = attr.get('mode', 'constant')
    if pad_mode == 'constant':
        res = np.pad(x, pads, pad_mode, constant_values=const_value)
    else:
        res = np.pad(x, pads, pad_mode)
    return res
