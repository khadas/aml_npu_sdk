from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from acuitylib.acuitylog import AcuityLog as al

import numpy as np  # type: ignore
from onnx.numpy_helper import to_array
from onnx import TensorProto

def _compute_dims(value):
    data_type = getattr(value, 'data_type', TensorProto.UNDEFINED)
    raw_data = getattr(value, 'raw_data')

    _type_map = {
        TensorProto.INT64:  8,
        TensorProto.INT32:  4,
        TensorProto.INT16:  2,
        TensorProto.UINT64: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.FLOAT:  4,
        TensorProto.FLOAT16: 2,
    }

    dims = []
    data_len = len(raw_data)
    if data_type in _type_map.keys():
        bit_width = _type_map[data_type]
        _dim = data_len // bit_width
        # Note: if dim == 1, it means a scale
        if _dim > 1:
            al.w("The TensorProto miss the dims, convert the constant data to a 1D numpy array.")
            dims = [_dim]
    return dims

def Constant(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    value = to_array(attr['value'])
    dims = getattr(attr['value'], 'dims', [])
    if len(value.shape) == 0 and dims == []:
        dims = _compute_dims(attr['value'])
        value = np.reshape(value, dims)
    return value
