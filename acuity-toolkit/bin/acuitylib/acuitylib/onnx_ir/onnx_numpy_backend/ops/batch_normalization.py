from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
  dims_x = len(x.shape)
  dim_ones = (1,) * (dims_x - 2)
  s = s.reshape(-1, *dim_ones)
  bias = bias.reshape(-1, *dim_ones)
  mean = mean.reshape(-1, *dim_ones)
  var = var.reshape(-1, *dim_ones)
  return s * (x - mean) / np.sqrt(var + epsilon) + bias

def BatchNormalization(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    scale = inputs[1]
    bias = inputs[2]
    mean = inputs[3]
    var = inputs[4]
    epsilon = attr.get('epsilon', 1e-05)
    # TODO: need support "momentum" attribute
    return _batchnorm_test_mode(x, scale, bias, mean, var, epsilon)
