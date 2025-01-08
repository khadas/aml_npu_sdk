from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def RandomNormalLike(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    shape = inputs[0].shape
    if 'seed' in attr:
        np.random.seed(attr['seed'])
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    return np.random.normal(loc=attr['mean'], scale=attr['scale'], size=shape).astype(TENSOR_TYPE_TO_NP_TYPE(attr['dtype']))
