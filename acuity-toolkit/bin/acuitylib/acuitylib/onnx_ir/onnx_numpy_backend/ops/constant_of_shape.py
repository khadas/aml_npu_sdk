from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def ConstantOfShape(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    import onnx.numpy_helper
    value_info = attr.get('value', None)
    x = inputs[0].astype(np.int32)
    if value_info != None:
        value = onnx.numpy_helper.to_array(value_info)
        dtype = value.dtype
    else:
        value = 0.0
        dtype = np.float32
    return np.ones(shape=x, dtype=dtype) * value
