from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def TopK(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
        sorted_indices = np.argsort(X, axis=axis)
        sorted_values = np.sort(X, axis=axis)
        if largest:
            sorted_indices = np.flip(sorted_indices, axis=axis)
            sorted_values = np.flip(sorted_values, axis=axis)
        topk_sorted_indices = np.take(sorted_indices, np.arange(k), axis=axis)
        topk_sorted_values = np.take(sorted_values, np.arange(k), axis=axis)
        return topk_sorted_values, topk_sorted_indices
    axis = attr.get('axis', -1)
    largest = attr.get('largest', 1)
    sorted = attr.get('sorted', 1)
    x = inputs[0]
    K = inputs[1]
    v,i = topk_sorted_implementation(x, K, axis, largest)
    return [v, i]
