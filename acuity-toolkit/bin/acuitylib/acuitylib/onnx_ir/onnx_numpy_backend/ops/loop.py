from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import math

def Loop(inputs, outputs, attr=None, op_version=11, scanner=None):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from acuitylib.onnx_ir.onnx_numpy_backend.onnx_backend import GraphRunner
    trip_count = math.inf if isinstance(inputs[0], str) and inputs[0] == '' else int(inputs[0])
    keep_running = True if isinstance(inputs[1], str) and inputs[1] == '' else bool(inputs[1])
    loop_name = attr['name']
    body_graph_name = loop_name + '_' + 'body'
    model_scanner = scanner.g_model
    body_graph_scanner = model_scanner.scanner(body_graph_name)
    '''
    Loop have op inputs and outputs, loop body have inputs and outputs.
    The relation is:
    Loop input contain : M[option], cond[option], v_initial[variadic]
        Loop body Graph input contain: count, cond, v_initial
        Loop body Graph output contain: cond, v_init, scan_out
    Loop output contain: v_init, scan_out
    '''

    run_counter = np.array(0).astype(np.int64)
    init_count = len(body_graph_scanner.in_tensors) - 2
    scan_out_cout = len(body_graph_scanner.out_tensors) - 1 - init_count
    scan_out = [[] for _ in range(scan_out_cout)]
    # will insert
    loop_inputs = [run_counter] + inputs[1:]
    init_to_final = [[] for _ in range(init_count)]
    while run_counter < trip_count and keep_running:
        loop_outputs = GraphRunner().run(model_scanner.minor_version,
                                         body_graph_scanner,
                                         loop_inputs,
                                         context=scanner.runtime_tbl
                                         )
        keep_running = bool(loop_outputs[0])
        run_counter = run_counter + np.array(1).astype(np.int64)
        loop_inputs.clear()
        loop_inputs.append(run_counter)
        loop_inputs.append(keep_running)
        loop_inputs.extend(loop_outputs[1:1+init_count])
        for index, scan_out_id in enumerate(range(1 + init_count, 1 + init_count + scan_out_cout)):
            scan_out[index].append(loop_outputs[scan_out_id])
        for index in range(init_count):
            init_to_final[index] = loop_outputs[1+index]

    res = init_to_final.copy()
    for id in range(scan_out_cout):
        if len(scan_out[id]) > 0:
            res.append(np.stack(scan_out[id]))
        else:
            res.append('')
    return res[0] if len(outputs) == 1 else res
