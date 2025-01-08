from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def RandomUniform(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    if 'seed' in attr:
        np.random.seed(attr['seed'])
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    return np.random.uniform(low=attr['low'], high=attr['high'], size=attr['shape']).astype(
        TENSOR_TYPE_TO_NP_TYPE(attr['dtype']))

