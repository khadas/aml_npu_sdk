from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def MeanVarianceNormalization(inputs, outputs, attr=None, op_version=11):
    # type: (np.ndarray, list, dict, int) -> np.ndarray
    input_data = inputs[0]
    axes = tuple(attr.get('axes', [0, 2, 3]))
    data_mean = np.mean(input_data, axis=axes, keepdims=1)
    data_mean_squared = np.power(data_mean, 2)
    data_squared = np.power(input_data, 2)
    data_squared_mean = np.mean(data_squared, axis=axes, keepdims=1)
    std = np.sqrt(data_squared_mean - data_mean_squared)
    res = (input_data - data_mean) / (std + 1e-9)
    return res
