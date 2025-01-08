from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Cast(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from onnx import TensorProto
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    if attr.get('to') == int(TensorProto.STRING):
        # Converting input to str, then give it np.object dtype for generating script
        x = inputs[0]
        shape = x.shape
        ss = []
        for i in x.flatten():
            s = str(i).encode('utf-8')
            su = s.decode('utf-8')
            ss.append(su)
        output = np.array(ss).astype(np.object).reshape(shape)
        return output
    return inputs[0].astype(TENSOR_TYPE_TO_NP_TYPE[attr.get('to')])
