from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def EyeLike(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    from onnx import TensorProto
    shape = inputs[0].shape
    return np.eye(shape[0], shape[1], k=attr.get('k', 0), dtype=TENSOR_TYPE_TO_NP_TYPE[attr.get('dtype', TensorProto.INT32)])
