from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Slice(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    in_count = len(inputs)
    data = inputs[0]
    if isinstance(data, np.ndarray) == False:
        data = np.array(data)
    if op_version < 10:
        axes = attr.get('axes', None)
        ends = attr['ends']
        starts = attr['starts']
        steps = None
    else:
        starts = inputs[1]
        ends = inputs[2]
        axes = list(inputs[3]) if in_count >= 4 else None
        steps = list(inputs[4]) if in_count >= 5 else None

    rank = data.ndim
    if axes == None:
        axes = [i for i in range(len(starts))]
    axes = [a if a >= 0 else rank + a for a in axes]
    slice_arg = list()
    for r in range(rank):
        if isinstance(axes, list) and r in axes:
            index = axes.index(r)
            if isinstance(steps, list):
                slice_arg.append(str(int(starts[index])) + ':' + str(int(ends[index])) + ':' + str(int(steps[index])))
            else:
                slice_arg.append(str(int(starts[index])) + ':' + str(int(ends[index])))
        elif isinstance(axes, list) and r not in axes:
            slice_arg.append(':')
        elif axes == None:
            slice_arg.append(str(int(starts[r])) + ':' + str(int(ends[r])))
    slice_arg_str = ','.join(slice_arg)
    res = eval('data[{}]'.format(slice_arg_str))
    return res
