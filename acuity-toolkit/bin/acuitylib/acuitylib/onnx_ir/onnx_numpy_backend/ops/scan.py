from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import copy

def Scan(inputs, outputs, attr=None, op_version=11, scanner=None):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from acuitylib.onnx_ir.onnx_numpy_backend.onnx_backend import GraphRunner
    num_scan_inputs = attr['num_scan_inputs']
    M = num_scan_inputs
    N = len(inputs) - M
    K = len(outputs) - N
    scan_input_axes = attr.get('scan_input_axes', [0] * M)
    scan_input_directions = attr.get('scan_input_directions', [0] * M)
    scan_output_axes = attr.get('scan_output_axes', [0] * N)
    scan_output_directions = attr.get('scan_output_directions', [0] * N)
    loop_name = attr['name']
    body_graph_name = loop_name + '_' + 'body'
    model_scanner = scanner.g_model
    body_graph_scanner = model_scanner.scanner(body_graph_name)
    '''
    Scan have op inputs and outputs, Scan body have inputs and outputs.
    The relation is:
    Scan input contain : state_variables[N], scan_inputs[M]
        Loop body Graph input contain: state_variables[N], scan_inputs[M]
        Loop body Graph output contain: state_variables[N], scan_outputs[K]
    Loop output contain: state_variables[N], scan_outputs[K]
    '''

    scan_out = [[] for _ in range(K)]
    # will insert
    init_N = inputs[:N]
    scan_inputs = inputs[N:]
    sequence_length = scan_inputs[0].shape[scan_input_axes[0]]
    loop_inputs = copy.copy(init_N)
    for t in range(sequence_length):
        for s in range(M):
            data = scan_inputs[s]
            axis = scan_input_axes[s]
            sections = data.shape[axis]
            the_nth_input = np.split(data, axis=axis, indices_or_sections=sections)[t]
            loop_inputs.append(the_nth_input)
        # TODO: the direction didn't consider here, should consider it when some case use it.

        loop_outputs = GraphRunner().run(model_scanner.minor_version,
                                         body_graph_scanner,
                                         loop_inputs,
                                         context=scanner.runtime_tbl
                                         )
        loop_inputs.clear()
        init_N = loop_outputs[0:N]
        loop_inputs.extend(copy.copy(init_N))

        for id in range(K):
            scan_out[id].append(loop_outputs[N + id])

    res = list()
    for n in range(N):
        res.append(init_N[0].reshape(inputs[n].shape))
    for id in range(K):
        if len(scan_out[id]) > 0:
            res.append(np.concatenate(scan_out[id], axis=scan_output_axes[id]))
        else:
            res.append('')
    return res[0] if len(outputs) == 1 else res
