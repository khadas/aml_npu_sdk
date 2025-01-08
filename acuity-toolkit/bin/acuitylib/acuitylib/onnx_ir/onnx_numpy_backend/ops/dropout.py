from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def Dropout(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    if len(outputs) == 1:
        return inputs[0]
    else:
        return (inputs[0], np.ones_like(inputs[0]).astype(np.bool))
