from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ReverseSequence(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    data = inputs[0]
    sequence_lens = inputs[1]
    batch_axis = attr.get('batch_axis', 1)
    time_axis = attr.get('time_axis', 0)
    if time_axis == 0:
        data = data.transpose()
    res_list = list()
    for i in range(len(sequence_lens)):
        seq_data = data[i]
        rev_len = sequence_lens[i]
        item = [d for d in reversed(seq_data[0:rev_len])] + [d for d in seq_data[rev_len:]]
        res_list.append(item)

    ret = np.array(res_list, dtype=data.dtype)
    if time_axis == 0:
        ret = ret.transpose()
    return ret
