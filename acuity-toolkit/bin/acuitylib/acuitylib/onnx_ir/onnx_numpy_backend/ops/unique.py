from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Unique(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    return_index = False
    return_inverse = False
    return_counts = False
    output_len = len(outputs)
    if output_len > 1:
        return_index = True
    if output_len > 2:
        return_inverse = True
    if output_len > 3:
        return_counts = True

    sort = attr.get('sorted', 1) == 1
    axis = attr.get('axis', None)
    if axis != None:
        return np.unique(inputs[0], return_index, return_inverse, return_counts, axis=axis)
    else:
        x = inputs[0].copy()
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)

        if sort == False:
            # prepare index mapping from sorted to unsorted
            argsorted_indices = np.argsort(indices)
            inverse_indices_map = {i: si for i, si in zip(argsorted_indices, np.arange(len(argsorted_indices)))}

            y = np.take(x, indices, axis=0)
            indices = indices[argsorted_indices]
            inverse_indices = np.asarray([inverse_indices_map[i] for i in inverse_indices], dtype=np.int64)
            counts = counts[argsorted_indices]
        return_list =[y]
        if output_len > 1:
            return_list.append(indices)
        if output_len > 2:
            return_list.append(inverse_indices)
        if output_len > 3:
            return_list.append(counts)
        return return_list
