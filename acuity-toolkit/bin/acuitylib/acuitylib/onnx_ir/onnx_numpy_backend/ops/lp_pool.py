from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def LpPool(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from .max_pool import get_output_shape, get_pad_shape, pool
    x = inputs[0]
    x_shape = np.shape(x)
    spatial_rank = len(x_shape) - 2
    kernel_shape = attr['kernel_shape']
    strides = attr.get('strides', [1]*spatial_rank)
    dilations = [1] * spatial_rank
    ceil_mode = attr.get('ceil_mode', 0) == 1
    spatial_pad = attr.get('pads', [0]*2*spatial_rank)
    out_spatial_shape = get_output_shape(attr.get('auto_pad', 'VALID', ), x_shape[2:], kernel_shape, strides,
                                         dilations=dilations, ceil_mode=ceil_mode, spatial_pad=spatial_pad)
    spatial_pads = get_pad_shape(attr.get('auto_pad', 'VALID'), x_shape[2:], kernel_shape, strides, dilations,
                                 out_spatial_shape, spatial_pad)
    pads = [(0, 0), (0, 0)] + spatial_pads
    padded = np.pad(x, pads, mode='constant', constant_values=np.nan)
    res = pool(padded, x_shape, kernel_shape, strides, out_spatial_shape, spatial_pads, 'LP{}'.format(attr.get('p',2)))
    return res
