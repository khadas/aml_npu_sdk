from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def scatter_nd_impl(data, indices, updates):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray

    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

    # Compute output
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        # NOTE: The order of iteration in this loop is not specified.
        # In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
        # This ensures that the output value does not depend on the iteration order.
        output[tuple(indices[i].astype(np.int64))] = updates[i]
    return output

def ScatterND(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    data = inputs[0].copy()
    indices = inputs[1]
    updates = inputs[2]
    return scatter_nd_impl(data, indices, updates)
