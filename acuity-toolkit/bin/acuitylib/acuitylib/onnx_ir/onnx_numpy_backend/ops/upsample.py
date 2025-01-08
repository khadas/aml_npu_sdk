from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from acuitylib.acuitylog import AcuityLog as al

import numpy as np  # type: ignore
from acuitylib.xtf import xtf as tf

def Upsample(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    data = inputs[0]
    mode = attr.get('mode', 'nearest')
    if op_version <= 7:
        scale = attr['scales']
    else:
        scale = inputs[1]

    # This is a correction value to fix the accuracy issue of float64 to float32
    # eg: out = 35.0, in = 18.0, scale = 35.0 / 18.0 = 1.94444444
    #     but out = floor(1.9444444 * 18.9) = 34
    correction_value = 1e-5
    new_shape = np.floor(np.multiply(data.shape, scale) + correction_value).astype(np.int32)
    if 'infer_shape' in attr:
        return np.ones(new_shape, data.dtype)

    if mode == 'linear':
        mode = 'bilinear'

    x = inputs[0]
    spatial_rank = x.ndim - 2
    perm = [0] + [i + 2 for i in range(spatial_rank)] + [1]
    x = np.transpose(x, perm)
    new_spatial_shape = new_shape[2:]

    if mode == 'nearest':
        upsamp = tf.compat.v1.image.resize_nearest_neighbor(x, new_spatial_shape)
    elif mode == 'bilinear':
        upsamp = tf.compat.v1.image.resize_bilinear(x, new_spatial_shape)
    else:
        upsamp = tf.compat.v1.image.resize_bicubic(x, new_spatial_shape)

    perm = [0] + [spatial_rank + 1] + [i + 1 for i in range(spatial_rank)]
    res = tf.transpose(upsamp, perm)
    return res.numpy()
