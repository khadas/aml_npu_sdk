from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

def If(inputs, outputs, attr=None, op_version=11, scanner=None):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from acuitylib.onnx_ir.onnx_numpy_backend.onnx_backend import GraphRunner
    if inputs[0]:
        mode_scanner = scanner.model_scanner
        else_scanner = mode_scanner.scanner(attr['name'] + '_else_branch')
        if_ret = GraphRunner().run(mode_scanner.minor_version, else_scanner, inputs[1:], context=scanner.runtime_tbl)
    else:
        mode_scanner = scanner.model_scanner
        then_scanner = mode_scanner.scanner(attr['name'] + '_then_branch')
        if_ret = GraphRunner().run(mode_scanner.minor_version, then_scanner, inputs[1:], context=scanner.runtime_tbl)

    if len(outputs) == 1:
        return if_ret[0]
    else:
        return  if_ret