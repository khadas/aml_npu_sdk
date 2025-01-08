from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def GatherND(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    def gather_nd_impl(data, indices):
        # type: (np.ndarray, np.ndarray) -> np.ndarray

        # Note the data rank - will be reused multiple times later
        data_rank = len(data.shape)

        # Check input tensors' shape/rank condition
        assert indices.shape[-1] <= data_rank

        # Compute output of the op as below
        # Compute shape of output array
        output_shape = list(indices.shape)[:-1] if (indices.shape[-1] == data_rank) else list(indices.shape)[
                                                                                         :-1] + list(data.shape)[
                                                                                                indices.shape[-1]:]

        # Placeholder for output data
        output_data_buffer = []

        # Flatten 'indices' to 2D array
        reshaped_indices = indices.reshape(-1, indices.shape[-1])

        # gather each scalar value from 'data'
        for outer_dim in range(reshaped_indices.shape[0]):
            gather_index = tuple(reshaped_indices[outer_dim])
            output_data_buffer.append(data[gather_index])
        return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)

    data = inputs[0]
    indices = inputs[1]
    return gather_nd_impl(data, indices)
