from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

# The below ScatterElements' numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def scatter_elements(data, indices, updates, axis=0):  # type: ignore
    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1:]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced indices for scattering of updates param. in data
    idx = [[unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]] for i in range(indices.shape[axis])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape)))

    scattered = np.copy(data)
    scattered[tuple(idx)] = updates[tuple(updates_idx)]
    return scattered

def ScatterElements(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    data = inputs[0].copy()
    indices = inputs[1]
    updates = inputs[2]
    axis = attr.get('axis', 0)
    if axis < 0:
        axis = data.ndim + axis
    return scatter_elements(data, indices, updates, axis=axis)


