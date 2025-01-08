from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Split(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    indices_or_sections = None
    if op_version < 13:
        indices_or_sections = attr.get('split', None)
    else:
        if len(inputs) == 2:
            if inputs[1] == '':
                indices_or_sections = None
            elif inputs[1].ndim == 1:
                indices_or_sections = inputs[1]
    axis = attr.get('axis', 0)

    if indices_or_sections is None:
        s = x.shape[axis]
        out_len = len(outputs)
        indices_or_sections = out_len
        remainder = s % indices_or_sections
        if remainder > 0:
            big_part_size = int(np.ceil(s / indices_or_sections))
            indices_or_sections = [big_part_size]*(out_len-1)
            small_part_size = s - big_part_size * (out_len-1)
            indices_or_sections.append(small_part_size)
        else:
            indices_or_sections = [int(s / out_len)] * out_len

    if isinstance(indices_or_sections, list):
        indices_or_sections= np.cumsum(indices_or_sections)
    return np.split(x, indices_or_sections=indices_or_sections, axis=axis)
