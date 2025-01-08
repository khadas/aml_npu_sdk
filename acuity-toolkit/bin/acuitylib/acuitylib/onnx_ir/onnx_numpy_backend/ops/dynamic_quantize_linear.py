from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def DynamicQuantizeLinear(inputs, outputs, attr=None, op_version=11):
    # type: (np.ndarray, list, dict, int) -> Tuple[Any, ...]
    X = inputs[0]
    x_min = np.minimum(0, np.min(X))
    x_max = np.maximum(0, np.max(X))
    Y_Scale = np.float32((x_max - x_min) / (255 - 0))
    Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
    Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)
    return (Y, Y_Scale, Y_ZeroPoint)
