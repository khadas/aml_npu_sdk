import json
import sys
import os

import dill

#CAUTION: please do not refer to acuitylib's internal module in this file

ruler_list = list()

def rule_pyfunc_def(func):
    def _wrap_func(*args, **kwargs):
        src = dill.source.getsource(func)
        src = src.replace('\r\n', '\n')
        src = src.replace('@rule_pyfunc_def\n', '')
        src = src.split('\n')
        return ['__rule_func_additional_args = ' + json.dumps(kwargs)] + src if len(kwargs) > 0 else src
    return _wrap_func

#support build-in functions:
'''
    def have_const_in_inputs(self, tensor):
    def input_port_is_const(self, tensor, pt):
    self.tensor_is_scalar(self, tensor):
    def shape_pick(self, tensor):
    def attr_pick(self, tensor, key, default=0):
    def array_layout(self, array, layout):
    def squeeze_shapes(self, squeeze_dims, input_shape):
    def split_slice(self, split_dims, split_shape):
    def tensor_to_numpy(self, tensor_name, trans=None):
    def tf_type_enum_to_ac_type(self, tf_type)
'''

@rule_pyfunc_def
def r_fc_tensor_dot_rule_get_param_bias(self, node, tensor):
    return True

@rule_pyfunc_def
def r_fc_tensor_dot_rule_get_param_axis(self, node, tensor, reshape_param, transpose_out):
    import functools
    shape_before_matmul = self.tensor_to_numpy(tensor[reshape_param])
    input_shape = self.shape_pick(tensor[transpose_out])

    multiply = lambda x, y: x * y
    positive_pos = -1
    positive_pos_actural_value = -1
    for i in range(len(shape_before_matmul)):
        if shape_before_matmul[i] == -1:
            positive_pos = i
            break
    if positive_pos > -1:
        positive_pos_actural_value = functools.reduce(multiply, input_shape)
        for i in range(len(shape_before_matmul)):
            if shape_before_matmul[i] != -1:
                positive_pos_actural_value = positive_pos_actural_value / shape_before_matmul[i]
    if positive_pos_actural_value > 0:
        shape_before_matmul[positive_pos] = positive_pos_actural_value

    axes = []
    for i in range(1, len(input_shape)):
        first = input_shape[:i]
        second = input_shape[i:]
        first = functools.reduce(multiply, first)
        second = functools.reduce(multiply, second)
        if first == shape_before_matmul[0] and second == shape_before_matmul[1]:
            axes.append(i)

    if len(axes) == 0:
        raise ValueError("Cannot find axis from {} to {}".format(input_shape, shape_before_matmul))
    elif len(axes) > 1:
        print("Warning: Find multiple axis from {} to {}".format(input_shape, shape_before_matmul))

    if len(axes) == 0 or len(axes) > 1:
        return -1
    else:
        return axes[0]

@rule_pyfunc_def
def r_fc_pre_condition(self, node, tensor, reshape_param, transpose_out, add_in1):
    pre1 = len(self.tensor_to_numpy(tensor[reshape_param]).tolist()) == 2 and \
           len(self.shape_pick(tensor[add_in1])) < 2 and \
           self.attr_pick(node['MatMul'], 'transpose_a', False) == False

    import functools
    shape_before_matmul = self.tensor_to_numpy(tensor[reshape_param])
    input_shape = self.shape_pick(tensor[transpose_out])

    multiply = lambda x, y: x * y
    positive_pos = -1
    positive_pos_actural_value = -1
    for i in range(len(shape_before_matmul)):
        if shape_before_matmul[i] == -1:
            positive_pos = i
            break
    if positive_pos > -1:
        positive_pos_actural_value = functools.reduce(multiply, input_shape)
        for i in range(len(shape_before_matmul)):
            if shape_before_matmul[i] != -1:
                positive_pos_actural_value = positive_pos_actural_value / shape_before_matmul[i]
    if positive_pos_actural_value > 0:
        shape_before_matmul[positive_pos] = positive_pos_actural_value

    axes = []
    for i in range(1, len(input_shape)):
        first = input_shape[:i]
        second = input_shape[i:]
        first = functools.reduce(multiply, first)
        second = functools.reduce(multiply, second)
        if first == shape_before_matmul[0] and second == shape_before_matmul[1]:
            axes.append(i)

    if len(axes) == 0:
        print("Warning: Axis pre-condition checking cannot find axis from {} to {}"
              .format(input_shape, shape_before_matmul))
    elif len(axes) > 1:
        print("Warning: Axis pre-condition checking find multiple axis from {} to {}"
              .format(input_shape, shape_before_matmul))

    pre2 = False
    if len(axes) == 0 or len(axes) > 1:
        pre2 = False
    else:
        pre2 = True

    return pre1 and pre2

@rule_pyfunc_def
def r_fc_tensor_dot_rule_get_weight(self, node, tensor):
    return self.tensor_to_numpy(tensor['C_2:out0'])

@rule_pyfunc_def
def r_fc_tensor_dot_rule_get_bias(self, node, tensor):
    return self.tensor_to_numpy(tensor['C:out0'])

@rule_pyfunc_def
def r_fc_tensor_dot_rule_pre_condition(self, node, tensor):
    return True

@rule_pyfunc_def
def r_conv2d_2_conv1d_pre_condition(self, node, tensor, conv2d_weight):
    #FIX ME: maybe we need more strict check here, such as check input is 1xN or Nx1
    return self.shape_pick(tensor[conv2d_weight])[0] == 1 \
        or self.shape_pick(tensor[conv2d_weight])[1] == 1

@rule_pyfunc_def
def r_conv2d_2_conv1d_ksize(self, node, tensor, conv2d_weight):
    return max(self.shape_pick(tensor[conv2d_weight])[0], self.shape_pick(tensor[conv2d_weight])[1])

@rule_pyfunc_def
def r_conv2d_2_conv1d_dilation(self, node, tensor):
    dilations = self.attr_pick(node['Conv'], 'dilations', [1,1,1,1])
    if isinstance(dilations, int):
        dilations = [1, dilations, dilations, 1]
    else:
        if len(dilations) == 1:
            dilations = [1, dilations[0], dilations[0], 1]
        elif len(dilations) ==2:
            dilations = [1, dilations[0], dilations[1], 1]
        elif len(dilations) == 4:
            pass
        else:
            print("Error: get unexpected dilations '{}' from conv2d".format(dilations))
            exit(-1)
    return [dilations[0], max(dilations[1], dilations[2]), dilations[3]]

@rule_pyfunc_def
def r_conv2d_2_conv1d_stride(self, node, tensor):
    strides = self.attr_pick(node['Conv'], 'strides', [1,1,1,1])
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]
    else:
        if len(strides) == 1:
            strides = [1, strides[0], strides[0], 1]
        elif len(strides) ==2:
            strides = [1, strides[0], strides[1], 1]
        elif len(strides) == 4:
            pass
        else:
            print("Error: get unexpected strides '{}' from conv2d".format(strides))
            exit(-1)
    return max(strides[1], strides[2])

r_variable = {
"ruler_name": "const",
"src_ops_alias": ["C"],
"src_inter_flow": [],
"src_in_anchor": [],
"src_out_tensor": ["C:out0"],
"acu_lys_alias": ["variable"],
"src_acu_in_tensor_map": [],
"src_acu_out_tensor_map": [["C:out0", "variable:out0"]],
"param_map": {"variable": {'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['C:out0'])"]}},
"blob_map": {"variable": {'data':
                              ['CODE',
                               "np.array([self.tensor_to_numpy(tensor['C:out0'])], dtype=np.float32) "\
                               " if self.tensor_to_numpy(tensor['C:out0']).shape == ()"\
                               "else self.tensor_to_numpy(tensor['C:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_variable)

r_rsp_mm_badd = {
"ruler_name": "rsp_fc_badd",
"src_ops_alias": ["BiasAdd", "MatMul", "C", "Reshape", "C_1", "C_2"],
"src_inter_flow": [["C_2:out0", "Reshape:in1"], ["C_1:out0", "MatMul:in1"], ["C:out0", "BiasAdd:in1"],
                   ["Reshape:out0", "MatMul:in0"], ["MatMul:out0", "BiasAdd:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "fullconnect:out0"]],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"],
                            'bias': ['BOOL', 'VALUE', True],
                            'axis': ['INT', 'PYFUNC',
                                     r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_2:out0',
                                    transpose_out='I:out0')],
                              }},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_fc_pre_condition(reshape_param='C_2:out0', transpose_out='I:out0', add_in1='C:out0')}
# ruler_list.append(r_rsp_mm_badd)

r_rsp_mm_add = {
"ruler_name": "rsp_fc_add",
"src_ops_alias": ["Add", "MatMul", "C", "Reshape", "C_1", "C_2"],
"src_inter_flow": [["C_2:out0", "Reshape:in1"], ["C_1:out0", "MatMul:in1"], ["C:out0", "Add:in1"],
                   ["Reshape:out0", "MatMul:in0"], ["MatMul:out0", "Add:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"],
                            'bias': ['BOOL', 'VALUE', True],
                            'axis': ['INT', 'PYFUNC',
                                       r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_2:out0',
                                                                           transpose_out='I:out0')],
                              }},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_fc_pre_condition(reshape_param='C_2:out0', transpose_out='I:out0', add_in1='C:out0')}
# ruler_list.append(r_rsp_mm_add)

r_rsp_mm_add_4d = {
"ruler_name": "fc+reshape",
"src_ops_alias": ["BiasAdd", "Reshape", "C", "MatMul", "C_1", "Reshape_1", "C_2", "C_3"],
"src_inter_flow": [["Reshape:out0", "BiasAdd:in0"], ["MatMul:out0", "Reshape:in0"], ["Reshape_1:out0", "MatMul:in0"],
                   ["C_1:out0", "Reshape:in1"], ["C_2:out0", "MatMul:in1"], ["C_3:out0", "Reshape_1:in1"],
                   ["C:out0", "BiasAdd:in1"]],
"src_in_anchor": [["I:out0", "Reshape_1:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["fullconnect", "reshape"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "reshape:out0"]],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"],
                                    'bias': ['BOOL', 'VALUE', True]},
              "reshape": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_1:out0'])"],}},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_2:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]},
             "reshape": {}},
"acu_inter_flow": [["fullconnect:out0", "reshape:in0"]],
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['C_3:out0']).tolist()) == 2 and " \
                "len(self.tensor_to_numpy(tensor['C_1:out0']).tolist()) == 4 and " \
                "len(self.shape_pick(tensor['C:out0'])) == 1 and "\
                "self.attr_pick(node['MatMul'], 'transpose_a', False) == False"}
#ruler_list.append(r_rsp_mm_add_4d)

r_mm_add= {
"ruler_name": "fullconnect",
"src_ops_alias": ["BiasAdd", "MatMul", "C", "C_1"],
"src_inter_flow": [["C_1:out0", "MatMul:in1"], ["C:out0", "BiasAdd:in1"], ["MatMul:out0", "BiasAdd:in0"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "fullconnect:out0"]],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"],
                            'bias': ['BOOL', 'VALUE', True],
                            }},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],
                        'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]
                        }},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['C:out0'])) < 2 and "\
                 "self.attr_pick(node['MatMul'], 'transpose_a', False) == False"}
ruler_list.append(r_mm_add)

r_mm ={
"ruler_name": "matmul_fc",
"src_ops_alias": ["MatMul", "C"],
"src_inter_flow": [["C:out0", "MatMul:in1"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["MatMul:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["MatMul:out0", "fullconnect:out0"]],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'], trans=[1, 0] "\
                                                         "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                         "else [0, 1]).shape[1]"],
                            'bias': ['BOOL', 'VALUE', False]}},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['MatMul'], 'transpose_a', False) == False"}
ruler_list.append(r_mm)

r_mm_1 = {
"ruler_name": "matmul_fc_1",
"src_ops_alias": ["MatMul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MatMul:in0"], ["I_1:out0", "MatMul:in1"]],
"src_out_tensor": ["MatMul:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map": [["I:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
"src_acu_out_tensor_map": [["MatMul:out0", "matmul:out0"]],
"acu_inter_flow": [],
"param_map": {"matmul": {'transpose_a': ['BOOL', 'CODE', "self.attr_pick(node['MatMul'], 'transpose_a', False)"],
                         'transpose_b': ['BOOL', 'CODE', "self.attr_pick(node['MatMul'], 'transpose_b', False)"],}},
"blob_map": {"matmul": {}},
"priority_tip": 0,
"pre_condition": None
}
ruler_list.append(r_mm_1)

r_mm_add_2_fc = {
"ruler_name": "fc_bias",
"src_ops_alias": ["Add", "MatMul", "C", "C_1"],
"src_inter_flow": [["MatMul:out0", "Add:in0"], ["C_1:out0", "MatMul:in1"], ["C:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"],
                            'bias': ['BOOL', 'VALUE', True]}},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],
                        'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['C:out0'])) < 2 and "\
                 "self.attr_pick(node['MatMul'], 'transpose_a', False) == False"}
ruler_list.append(r_mm_add_2_fc)

r_bmm = {
"ruler_name": "matmul",
"src_ops_alias": ["BatchMatMul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "BatchMatMul:in0"], ["I_1:out0", "BatchMatMul:in1"]],
"src_out_tensor": ["BatchMatMul:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map": [["I:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
"src_acu_out_tensor_map": [["BatchMatMul:out0", "matmul:out0"]],
"param_map": {"matmul": {'transpose_a': ['INT', 'CODE', "self.attr_pick(node['BatchMatMul'], 'adj_x', False)"],
                         'transpose_b': ['INT', 'CODE', "self.attr_pick(node['BatchMatMul'], 'adj_y', False)"],}},
"blob_map": {"matmul": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_bmm)

r_bv2mm = {
"ruler_name": "batchmatmulv2_to_matmul",
"src_ops_alias": ["BatchMatMulV2"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "BatchMatMulV2:in0"], ["I_1:out0", "BatchMatMulV2:in1"]],
"src_out_tensor": ["BatchMatMulV2:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map": [["I:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
"src_acu_out_tensor_map": [["BatchMatMulV2:out0", "matmul:out0"]],
"acu_inter_flow": [],
"param_map": {"matmul": {'transpose_a': ['BOOL', 'CODE', "self.attr_pick(node['BatchMatMulV2'], 'adj_x', False)"],
                         'transpose_b': ['BOOL', 'CODE', "self.attr_pick(node['BatchMatMulV2'], 'adj_y', False)"],}},
"blob_map": {"matmul": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_bv2mm)

r_relu = {
"ruler_name": "relu",
"src_ops_alias": ["Relu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Relu:in0"]],
"src_out_tensor": ["Relu:out0"],
"acu_lys_alias": ["relu"],
"src_acu_in_tensor_map": [["I:out0", "relu:in0"]],
"src_acu_out_tensor_map": [["Relu:out0", "relu:out0"]],
"param_map": {"relu": {}},
"blob_map": {"relu": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_relu)

r_elu = {
"ruler_name": "elu",
"src_ops_alias": ["Elu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Elu:in0"]],
"src_out_tensor": ["Elu:out0"],
"acu_lys_alias": ["elu"],
"src_acu_in_tensor_map": [["I:out0", "elu:in0"]],
"src_acu_out_tensor_map": [["Elu:out0", "elu:out0"]],
"param_map": {"elu": {}},
"blob_map": {"elu": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_elu)

r_relu_quant = {
"ruler_name": "quant_relu",
"src_ops_alias": ["FakeQuantWithMinMaxVars", "Relu", "C", "C_1"],
"src_inter_flow": [["Relu:out0", "FakeQuantWithMinMaxVars:in0"], ["C_1:out0", "FakeQuantWithMinMaxVars:in2"],
                   ["C:out0", "FakeQuantWithMinMaxVars:in1"]],
"src_in_anchor": [["I:out0", "Relu:in0"]],
"src_out_tensor": ["FakeQuantWithMinMaxVars:out0"],
"acu_lys_alias": ["relu"],
"src_acu_in_tensor_map": [["I:out0", "relu:in0"]],
"src_acu_out_tensor_map": [["FakeQuantWithMinMaxVars:out0", "relu:out0"]],
"param_map": {"relu": {}},
"blob_map": {"relu": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_relu_quant)

r_sigmoid = {
"ruler_name": "sigmoid",
"src_ops_alias": ["Sigmoid"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sigmoid:in0"]],
"src_out_tensor": ["Sigmoid:out0"],
"acu_lys_alias": ["sigmoid"],
"src_acu_in_tensor_map": [["I:out0", "sigmoid:in0"]],
"src_acu_out_tensor_map": [["Sigmoid:out0", "sigmoid:out0"]],
"param_map": {"sigmoid": {}},
"blob_map": {"sigmoid": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_sigmoid)


# r_relu_fk = dict()
# r_relu_fk['src_ops_alias'] = ['Relu', 'FakeQuantWithMinMaxVars']
# r_relu_fk['acu_lys_alias'] = ['relu']
# ruler_list.append(r_relu_fk)
r_relu6 ={
"ruler_name": "relu6",
"src_ops_alias": ["Relu6"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Relu6:in0"]],
"src_out_tensor": ["Relu6:out0"],
"acu_lys_alias": ["relun"],
"src_acu_in_tensor_map": [["I:out0", "relun:in0"]],
"src_acu_out_tensor_map": [["Relu6:out0", "relun:out0"]],
"param_map": {"relun": {'relu_clamp_top':['INT', 'VALUE', 6],}},
"blob_map": {"relun": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_relu6)

r_relu6_quant = {
"ruler_name": "quant_relu6",
"src_ops_alias": ["FakeQuantWithMinMaxVars", "Relu6", "C", "C_1"],
"src_inter_flow": [["Relu6:out0", "FakeQuantWithMinMaxVars:in0"], ["C_1:out0", "FakeQuantWithMinMaxVars:in2"],
                   ["C:out0", "FakeQuantWithMinMaxVars:in1"]],
"src_in_anchor": [["I:out0", "Relu6:in0"]],
"src_out_tensor": ["FakeQuantWithMinMaxVars:out0"],
"acu_lys_alias": ["relun"],
"src_acu_in_tensor_map": [["I:out0", "relun:in0"]],
"src_acu_out_tensor_map": [["FakeQuantWithMinMaxVars:out0", "relun:out0"]],
"param_map": {"relun": {'relu_clamp_top':['INT', 'VALUE', 6],}},
"blob_map": {"relun": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_relu6_quant)

r_reluN ={
"ruler_name": "relun",
"src_ops_alias": ["Minimum", "Relu", "C"],
"src_inter_flow": [["Relu:out0", "Minimum:in0"], ["C:out0", "Minimum:in1"]],
"src_in_anchor": [["I:out0", "Relu:in0"]],
"src_out_tensor": ["Minimum:out0"],
"acu_lys_alias": ["relun"],
"src_acu_in_tensor_map": [["I:out0", "relun:in0"]],
"src_acu_out_tensor_map": [["Minimum:out0", "relun:out0"]],
"param_map": {"relun": {'relu_clamp_top':['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()[0]"]}},
"blob_map": {"relun": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_reluN)

r_tfleakrelu ={
"ruler_name": "leakyrelu",
"src_ops_alias": ["Maximum", "Mul", "C"],
"src_inter_flow": [["C:out0", "Mul:in0"], ["Mul:out0", "Maximum:in0"]],
"src_in_anchor": [["I:out0", "Maximum:in1"], ["I:out0", "Mul:in1"]],
"src_out_tensor": ["Maximum:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["Maximum:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {'leaky_ratio':['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"blob_map": {"leakyrelu": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['C:out0']).tolist()) == 1 and "
                 "isinstance(self.tensor_to_numpy(tensor['C:out0']).tolist()[0], float)"}
ruler_list.append(r_tfleakrelu)

r_gelu_no_approximate = {
"ruler_name": "gelu_no_approximate",
"src_ops_alias": ["Mul", "Mul_1", "C", "Add", "C_1", "Erf", "RealDiv", "C_2"],
"src_inter_flow": [["Mul_1:out0", "Mul:in1"], ["C:out0", "Mul_1:in0"], ["Add:out0", "Mul_1:in1"],
                   ["C_1:out0", "Add:in0"], ["Erf:out0", "Add:in1"], ["RealDiv:out0", "Erf:in0"],
                   ["C_2:out0", "RealDiv:in1"]],
"src_in_anchor": [["I:out0", "RealDiv:in0"], ["I:out0", "Mul:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate':['BOOL', 'VALUE', False]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gelu_no_approximate)

r_gelu_no_approximate_tf_2_50 = {
"ruler_name": "gelu_no_approximate_tf_2_50",
"src_ops_alias": ["Mul", "Mul_1", "AddV2", "C", "C_1", "Erf", "RealDiv", "C_2"],
"src_inter_flow": [["Mul_1:out0", "Mul:in0"], ["AddV2:out0", "Mul:in1"], ["C:out0", "Mul_1:in0"],
    ["C_1:out0", "AddV2:in0"], ["Erf:out0", "AddV2:in1"], ["RealDiv:out0", "Erf:in0"],
    ["C_2:out0", "RealDiv:in1"]],
"src_in_anchor": [["I:out0", "Mul_1:in1"], ["I:out0", "RealDiv:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate':['BOOL', 'VALUE', False]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gelu_no_approximate_tf_2_50)

r_gelu_with_approximate = {
    "ruler_name": "r_gelu_with_approximate",
    "src_ops_alias": ["Mul", "Mul_1", "Add", "C", "C_1", "Tanh", "Mul_2", "C_2", "Add_1", "Mul_3",
                      "C_3", "Pow", "C_4"],
    "src_inter_flow": [["Mul_1:out0", "Mul:in0"], ["Add:out0", "Mul:in1"], ["C:out0", "Mul_1:in0"],
                       ["C_1:out0", "Add:in0"], ["Tanh:out0", "Add:in1"], ["Mul_2:out0", "Tanh:in0"],
                       ["C_2:out0", "Mul_2:in0"], ["Add_1:out0", "Mul_2:in1"], ["Mul_3:out0", "Add_1:in1"],
                       ["C_3:out0", "Mul_3:in0"], ["Pow:out0", "Mul_3:in1"], ["C_4:out0", "Pow:in1"]],
    "src_in_anchor": [["I:out0", "Mul_1:in1"], ["I:out0", "Pow:in0"], ["I:out0", "Add_1:in0"]],
    "src_out_tensor": ["Mul:out0"],
    "acu_lys_alias": ["gelu"],
    "src_acu_in_tensor_map": [["I:out0", "gelu:in0"]],
    "src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
    "acu_inter_flow": [],
    "param_map": {"gelu": {'approximate':['BOOL', 'VALUE', True]}},
    "blob_map": {"gelu": {}},
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_gelu_with_approximate)

r_gelu_with_approximate_2 = {
"ruler_name": 'r_gelu_with_approximate_2',
"src_ops_alias": ["Mul", "Mul_1", "C", "AddV2", "C_1", "Tanh", "Mul_2", "C_2",
                  "AddV2_1", "Mul_3", "C_3", "Pow", "C_4"],
"src_inter_flow": [["Mul_1:out0", "Mul:in1"], ["C:out0", "Mul_1:in0"], ["AddV2:out0", "Mul_1:in1"],
                   ["C_1:out0", "AddV2:in0"], ["Tanh:out0", "AddV2:in1"], ["Mul_2:out0", "Tanh:in0"],
                   ["C_2:out0", "Mul_2:in0"], ["AddV2_1:out0", "Mul_2:in1"], ["Mul_3:out0", "AddV2_1:in1"],
                   ["C_3:out0", "Mul_3:in0"], ["Pow:out0", "Mul_3:in1"], ["C_4:out0", "Pow:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Pow:in0"], ["I:out0", "AddV2_1:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate': ['BOOL', 'VALUE', True]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gelu_with_approximate_2)

r_gelu_with_approximate_3 = {
"ruler_name": "r_gelu_with_approximate_3",
"src_ops_alias": ["Mul", "Mul_1", "C", "Add", "C_1", "Tanh", "Mul_2",
                  "C_2", "Add_1", "Mul_3", "C_3", "Pow", "C_4"],
"src_inter_flow": [["Mul_1:out0", "Mul:in1"], ["C:out0", "Mul_1:in0"],
                   ["Add:out0", "Mul_1:in1"], ["C_1:out0", "Add:in0"],
                   ["Tanh:out0", "Add:in1"], ["Mul_2:out0", "Tanh:in0"], ["C_2:out0", "Mul_2:in0"],
                   ["Add_1:out0", "Mul_2:in1"], ["Mul_3:out0", "Add_1:in1"], ["C_3:out0", "Mul_3:in0"],
                   ["Pow:out0", "Mul_3:in1"], ["C_4:out0", "Pow:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Pow:in0"], ["I:out0", "Add_1:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate': ['BOOL', 'VALUE', True]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gelu_with_approximate_3)

r_gelu_with_approximate_tf_2_50 = {
"ruler_name": "r_gelu_with_approximate_tf_2_50",
"src_ops_alias": ["Mul", "Mul_1", "AddV2", "C", "C_1", "Tanh", "Mul_2", "C_2", "AddV2_1", "Mul_3",
                  "C_3", "Pow", "C_4"],
"src_inter_flow": [["Mul_1:out0", "Mul:in0"], ["AddV2:out0", "Mul:in1"], ["C:out0", "Mul_1:in0"],
    ["C_1:out0", "AddV2:in0"], ["Tanh:out0", "AddV2:in1"], ["Mul_2:out0", "Tanh:in0"],
    ["C_2:out0", "Mul_2:in0"], ["AddV2_1:out0", "Mul_2:in1"], ["Mul_3:out0", "AddV2_1:in1"],
    ["C_3:out0", "Mul_3:in0"], ["Pow:out0", "Mul_3:in1"], ["C_4:out0", "Pow:in1"]],
"src_in_anchor": [["I:out0", "Mul_1:in1"], ["I:out0", "Pow:in0"], ["I:out0", "AddV2_1:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate':['BOOL', 'VALUE', True]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gelu_with_approximate_tf_2_50)

r_minimum ={
"ruler_name": "minimum",
"src_ops_alias": ["Minimum"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Minimum:in0"], ["I_1:out0", "Minimum:in1"]],
"src_out_tensor": ["Minimum:out0"],
"acu_lys_alias": ["minimum"],
"src_acu_in_tensor_map": [["I:out0", "minimum:in0"], ["I_1:out0", "minimum:in1"]],
"src_acu_out_tensor_map": [["Minimum:out0", "minimum:out0"]],
"param_map": {"minimum": {}},
"blob_map": {"minimum": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_minimum)

r_tfleakrelu_2 = {
"ruler_name": "leakyrelu_2",
"src_ops_alias": ["Maximum", "Mul", "C"],
"src_inter_flow": [["Mul:out0", "Maximum:in1"], ["C:out0", "Mul:in0"]],
"src_in_anchor": [["I:out0", "Maximum:in0"], ["I:out0", "Mul:in1"]],
"src_out_tensor": ["Maximum:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["Maximum:out0", "leakyrelu:out0"]],
"acu_inter_flow": [],
"param_map": {"leakyrelu": {'leaky_ratio':['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"blob_map": {"leakyrelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_tfleakrelu_2)

r_leaky_relu_3 = {
"ruler_name": "leakyrelu_3",
"src_ops_alias": ["LeakyRelu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LeakyRelu:in0"]],
"src_out_tensor": ["LeakyRelu:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["LeakyRelu:out0", "leakyrelu:out0"]],
"acu_inter_flow": [],
"param_map": {
             "leakyrelu": {'leaky_ratio': ['FLOAT', 'CODE', "self.attr_pick(node['LeakyRelu'], 'alpha')"]}},
"blob_map": {"leakyrelu": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_leaky_relu_3)

r_mulmax2leakyrelu = {
"ruler_name": "mulmax2leakyrelu",
"src_ops_alias": ["Maximum", "Mul", "C"],
"src_inter_flow": [["Mul:out0", "Maximum:in0"], ["C:out0", "Mul:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Maximum:in1"]],
"src_out_tensor": ["Maximum:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["Maximum:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {'leaky_ratio':['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"blob_map": {"leakyrelu": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mulmax2leakyrelu)
#Maximum:g_conv1_1/Maximum;Mul:g_conv1_1/mul;C:g_conv1_1/mul/y

r_abs_add_2_leakyrelu = {
"ruler_name": "abs_mul_2_leakeyrelu",
"src_ops_alias": ["Add", "Mul", "Mul_1", "C", "C_1", "Abs"],
"src_inter_flow": [["Abs:out0", "Mul_1:in1"], ["Mul_1:out0", "Add:in1"], ["Mul:out0", "Add:in0"],
                   ["C_1:out0", "Mul_1:in0"], ["C:out0", "Mul:in0"]],
"src_in_anchor": [["I:out0", "Mul:in1"], ["I:out0", "Abs:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {'leaky_ratio':
                                ['FLOAT',
                                 'CODE',
                                 "self.tensor_to_numpy(tensor['C:out0']) - self.tensor_to_numpy(tensor['C_1:out0'])"
                                 ],
                            }},
"blob_map": {"leakyrelu": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "math.isclose(self.tensor_to_numpy(tensor['C:out0']) + self.tensor_to_numpy(tensor['C_1:out0']), 1.0)"
}
ruler_list.append(r_abs_add_2_leakyrelu)

r_tanh ={
"ruler_name": "tanh",
"src_ops_alias": ["Tanh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Tanh:in0"]],
"src_out_tensor": ["Tanh:out0"],
"acu_lys_alias": ["tanh"],
"src_acu_in_tensor_map": [["I:out0", "tanh:in0"]],
"src_acu_out_tensor_map": [["Tanh:out0", "tanh:out0"]],
"blob_map": {"tanh": {}},
"param_map": {"tanh": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_tanh)

r_prelu = {
"ruler_name": "prelu",
"src_ops_alias": ["Add", "Relu", "Mul", "Mul_1", "C", "C_1", "Sub", "Abs"],
"src_inter_flow": [["Mul:out0", "Add:in1"], ["C_1:out0", "Mul_1:in0"], ["Abs:out0", "Sub:in1"],
                   ["Relu:out0", "Add:in0"], ["Mul_1:out0", "Mul:in0"], ["C:out0", "Mul:in1"],
                   ["Sub:out0", "Mul_1:in1"]],
"src_in_anchor": [["I:out0", "Sub:in0"], ["I:out0", "Abs:in0"], ["I:out0", "Relu:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "prelu:out0"]],
"param_map": {"prelu": {}},
"blob_map": {"prelu": {'a': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_prelu)

r_merge_to_prelu = {
"ruler_name": 'r_merge_to_prelu',
"src_ops_alias": ["Sub", "Relu", "Mul", "C", "Relu_1", "Neg"],
"src_inter_flow": [["Relu:out0", "Sub:in0"], ["Mul:out0", "Sub:in1"], ["C:out0", "Mul:in0"],
                   ["Relu_1:out0", "Mul:in1"],
    ["Neg:out0", "Relu_1:in0"]],
"src_in_anchor": [["I:out0", "Relu:in0"], ["I:out0", "Neg:in0"]],
"src_out_tensor": ["Sub:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["Sub:out0", "prelu:out0"]],
"acu_inter_flow": [],
"param_map": {"prelu": {}},
"blob_map": {"prelu": {'a': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"priority_tip": 0,
"pre_condition": None
}
ruler_list.append(r_merge_to_prelu)

r_conv_1d = {
"ruler_name": "conv1d",
"src_ops_alias": ["Squeeze", "Conv", "ExpandDims", "C", "C_1"],
"src_inter_flow": [["Conv:out0", "Squeeze:in0"], ["ExpandDims:out0", "Conv:in0"],
                   ["C:out0", "Conv:in1"], ["C_1:out0", "ExpandDims:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "conv1d:out0"]],
"acu_inter_flow": [],
"param_map": {"conv1d": {   'ksize': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[1]"],
                            'stride': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[3]"]}},
"blob_map": {"conv1d": {'weight': ['CODE', "np.squeeze(self.tensor_to_numpy(tensor['C:out0']), 0)"]}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv_1d)

r_conv_1d_bias = {
"ruler_name": "conv1d_bias",
"src_ops_alias": ["BiasAdd", "Squeeze", "C", "Conv", "ExpandDims", "C_1", "C_2"],
"src_inter_flow": [["Squeeze:out0", "BiasAdd:in0"], ["ExpandDims:out0", "Conv:in0"], ["C_2:out0", "ExpandDims:in1"],
                   ["C:out0", "BiasAdd:in1"], ["Conv:out0", "Squeeze:in0"], ["C_1:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "conv1d:out0"]],
"param_map": {"conv1d": {   'ksize': ['INT', 'PYFUNC', r_conv2d_2_conv1d_ksize(conv2d_weight='C_1:out0')],
                            'stride': ['INT', 'PYFUNC', r_conv2d_2_conv1d_stride()],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'dilation': ['INTS', 'PYFUNC', r_conv2d_2_conv1d_dilation()],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"]}},
"blob_map": {"conv1d": {'weight': ['CODE', "np.squeeze(self.tensor_to_numpy(tensor['C_1:out0']), 0)"],
                        'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_conv2d_2_conv1d_pre_condition(conv2d_weight='C_1:out0')}
ruler_list.append(r_conv_1d_bias)

r_conv ={
"ruler_name": "single_convolution",
"src_ops_alias": ["Conv", "C"],
"src_inter_flow": [["C:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'dilation': ['INTS', 'CODE', "self.attr_pick(node['Conv'], 'dilations', [1,1,1,1])"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[3]"]}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv)

r_pad_conv = {
"ruler_name": "pad_convolution",
"src_ops_alias": ["Conv", "Pad", "C", "C_1"],
"src_inter_flow": [["C:out0", "Conv:in1"], ["C_1:out0", "Pad:in1"], ["Pad:out0", "Conv:in0"]],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'pad_h': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])[1][0]"],
                            'pad_w': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])[2][0]"],
                            'pad_method': ['STRING', 'VALUE', "padding_const"],
                            'pad': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])[1:3].flatten()"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[3]"],}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Conv'], 'padding') == 'VALID'"}
ruler_list.append(r_pad_conv)

r_pad_conv_badd = {
"ruler_name": "pad_convolution_badd",
"src_ops_alias": ["BiasAdd", "Conv", "C", "Pad", "C_1", "C_2"],
"src_inter_flow": [["C_2:out0", "Pad:in1"], ["C:out0", "BiasAdd:in1"], ["Conv:out0", "BiasAdd:in0"],
                   ["C_1:out0", "Conv:in1"], ["Pad:out0", "Conv:in0"]],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'pad_h': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])[1][0]"],
                            'pad_w': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])[2][0]"],
                            'pad_method': ['STRING', 'VALUE', "padding_const"],
                            'pad': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])[1:3].flatten()"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[3]"],}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Conv'], 'padding') == 'VALID'"}
ruler_list.append(r_pad_conv_badd)

r_conv_badd = {
"ruler_name": "convolution_biasadd",
"src_ops_alias": ["BiasAdd", "Conv", "C", "C_1"],
"src_inter_flow": [["Conv:out0", "BiasAdd:in0"], ["C:out0", "BiasAdd:in1"], ["C_1:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[3]"],}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv_badd)

r_conv_badd_quant_w_quant_o = {
"ruler_name": "conv_badd_quant_w_quant_o",
"src_ops_alias": ["FakeQuantWithMinMaxVars", "BiasAdd", "C", "C_1", "Conv", "C_2", "FakeQuantWithMinMaxVars_1",
                  "C_3", "C_4", "C_5"],
"src_inter_flow": [["C_5:out0", "FakeQuantWithMinMaxVars_1:in2"], ["C_1:out0", "FakeQuantWithMinMaxVars:in2"],
                   ["FakeQuantWithMinMaxVars_1:out0", "Conv:in1"], ["C_2:out0", "BiasAdd:in1"],
                   ["C_4:out0", "FakeQuantWithMinMaxVars_1:in1"], ["C:out0", "FakeQuantWithMinMaxVars:in1"],
                   ["Conv:out0", "BiasAdd:in0"], ["C_3:out0", "FakeQuantWithMinMaxVars_1:in0"],
                   ["BiasAdd:out0", "FakeQuantWithMinMaxVars:in0"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["FakeQuantWithMinMaxVars:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["FakeQuantWithMinMaxVars:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[3]"],}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv_badd_quant_w_quant_o)

r_conv_badd_quant_w = {
"ruler_name": "conv_badd_quant_w",
"src_ops_alias": ["BiasAdd", "Conv", "C_2", "FakeQuantWithMinMaxVars_1",
                  "C_3", "C_4", "C_5"],
"src_inter_flow": [["C_5:out0", "FakeQuantWithMinMaxVars_1:in2"],
                   ["FakeQuantWithMinMaxVars_1:out0", "Conv:in1"], ["C_2:out0", "BiasAdd:in1"],
                   ["C_4:out0", "FakeQuantWithMinMaxVars_1:in1"],
                   ["Conv:out0", "BiasAdd:in0"], ["C_3:out0", "FakeQuantWithMinMaxVars_1:in0"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[3]"],}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv_badd_quant_w)


# r_conv_badd_scale = dict()
# r_conv_badd_scale['src_ops_alias'] = ['Conv2D', 'BiasAdd', 'Mul']
# r_conv_badd_scale['acu_lys_alias'] = ['convolution']
# r_conv_badd_scale['pre_condition'] =
# "len(tensor['Conv2D'].input) == 2 and self.input_port_is_const(tensor['Mul'], 'in0')"
# r_conv_badd_scale['src_internal_edges'] =
# [[['Conv2D', 'out0'], ['BiasAdd', 'in0']],[['BiasAdd', 'out0'], ['Mul', 'in1']]]
# r_conv_badd_scale['param_map'] = {'convolution':
#                            {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['Conv2D'].input[1])[0]"],
#                             'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['Conv2D'].input[1])[1]"],
#                             'stride_h':['INT', 'CODE', "self.attr_pick(tensor['Conv2D'], 'strides')[1]"],
#                             'stride_w':['INT', 'CODE', "self.attr_pick(tensor['Conv2D'], 'strides')[2]"],
#                             'padding': ['STRING', 'CODE', "self.attr_pick(tensor['Conv2D'], 'padding')"],
#                             'bias': ['BOOL', 'VALUE', True],
#                             'weights': ['INT', 'CODE', "self.shape_pick(tensor['Conv2D'].input[1])[3]"],
#                             }
#                     }
# r_conv_badd_scale['blob_map'] = {'convolution':
#{'weight': ['CODE', "self.tensor_to_numpy(tensor['Conv2D'].input[1]) * self.tensor_to_numpy(tensor['Mul'].input[0])"],
# 'bias': ['CODE', "self.tensor_to_numpy(tensor['BiasAdd'].input[1]) * self.tensor_to_numpy(tensor['Mul'].input[0])"]
# }
#                    }
# ruler_list.append(r_conv_badd_scale)

r_conv_add = {
"ruler_name": "conv_add",
"src_ops_alias": ["Add", "Conv", "C", "C_1"],
"src_inter_flow": [["C_1:out0", "Conv:in1"], ["C:out0", "Add:in1"], ["Conv:out0", "Add:in0"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[3]"],
                            }},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                           'bias': ['CODE', "np.reshape(self.tensor_to_numpy(tensor['C:out0']), [-1])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv_add)

r_quant_conv_add = {
"ruler_name": "conv_bias",
"src_ops_alias": ["Add", "Conv", "C", "FakeQuantWithMinMaxVars", "C_1", "C_2", "C_3"],
"src_inter_flow": [["C:out0", "Add:in1"], ["Conv:out0", "Add:in0"], ["FakeQuantWithMinMaxVars:out0", "Conv:in1"],
                   ["C_1:out0", "FakeQuantWithMinMaxVars:in0"], ["C_2:out0", "FakeQuantWithMinMaxVars:in1"],
                   ["C_3:out0", "FakeQuantWithMinMaxVars:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[3]"],}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                           'bias': ['CODE', "np.reshape(self.tensor_to_numpy(tensor['C:out0']), [-1])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_quant_conv_add)


#
# r_conv_add = dict()
# r_conv_add['src_ops_alias'] = ['Conv2D', 'BiasAdd_2', 'FakeQuantWithMinMaxVars_3']
# r_conv_add['acu_lys_alias'] = ['convolution_1']
# r_conv_add['pre_condition'] = "len(tensor['Conv2D'].input) == 2"
# r_conv_add['param_map'] = {'convolution':
#                            {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['Conv2D'].input[1])[0]"],
#                             'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['Conv2D'].input[1])[1]"],
#                             'stride_h':['INT', 'CODE', "self.attr_pick(tensor['Conv2D'], 'strides')[1]"],
#                             'stride_w':['INT', 'CODE', "self.attr_pick(tensor['Conv2D'], 'strides')[2]"],
#                             'padding': ['STRING', 'CODE', "self.attr_pick(tensor['Conv2D'], 'padding')"],
#                             'bias': ['BOOL', 'VALUE', True],
#                             'weights': ['INT', 'CODE', "self.shape_pick(tensor['Conv2D'].input[1])[3]"],
#                             }
#                     }
# r_conv_add['blob_map'] = {'convolution':
#                           {'weight': ['CODE', "self.tensor_to_numpy(tensor['Conv2D'].input[1])"],
#                            'bias': ['CODE', "self.tensor_to_numpy(tensor['BiasAdd_2'].input[1])"]
#                            }
#                    }
# ruler_list.append(r_conv_add)
r_dwconv = {
"ruler_name": "dwconv",
"src_ops_alias": ["DepthwiseConv2dNative", "C"],
"src_inter_flow": [["C:out0", "DepthwiseConv2dNative:in1"]],
"src_in_anchor": [["I:out0", "DepthwiseConv2dNative:in0"]],
"src_out_tensor": ["DepthwiseConv2dNative:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["DepthwiseConv2dNative:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'padding')"],
                            'pad': ['INTS', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'],\
                                'explicit_paddings', [0,0,0,0,0,0,0,0])"],
                            'dilation': ['INTS', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'],\
                                 'dilations', [1,1,1,1])"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights':
                                  ['INT',
                                   'CODE',
                                   "self.shape_pick(tensor['C:out0'])[2] * self.shape_pick(tensor['C:out0'])[3]"
                                   ],
                            'group_number': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[2]"]}},
"blob_map": {"convolution": {
'weight': ['CODE', "np.reshape("\
                   "self.tensor_to_numpy(tensor['C:out0']), "\
                   "[self.shape_pick(tensor['C:out0'])[0], self.shape_pick(tensor['C:out0'])[1], 1, -1])"
           ],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dwconv)

r_dwconv_quant_w_quant_o = {
"ruler_name": "dwconv_quant_w_quant_output",
"src_ops_alias": ["DepthwiseConv2dNative", "FakeQuantWithMinMaxVars", "C_1", "C_2", "C_3",
                  "FakeQuantWithMinMaxVars_1", "C_4", "C_5"],
"src_inter_flow": [["C_3:out0", "FakeQuantWithMinMaxVars:in2"],
                   ["C_1:out0", "FakeQuantWithMinMaxVars:in0"],
                   ["FakeQuantWithMinMaxVars:out0", "DepthwiseConv2dNative:in1"],
                   ["C_2:out0", "FakeQuantWithMinMaxVars:in1"],
                   ["DepthwiseConv2dNative:out0", "FakeQuantWithMinMaxVars_1:in0"],
                   ["C_4:out0", "FakeQuantWithMinMaxVars_1:in1"],
                   ["C_5:out0", "FakeQuantWithMinMaxVars_1:in2"]],
"src_in_anchor": [["I:out0", "DepthwiseConv2dNative:in0"]],
"src_out_tensor": ["FakeQuantWithMinMaxVars_1:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["FakeQuantWithMinMaxVars_1:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': [
                                'INT',
                                'CODE',
                                "self.shape_pick(tensor['C_1:out0'])[2] * self.shape_pick(tensor['C_1:out0'])[3]"
                            ],
                            'group_number': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[2]"]}},
"blob_map": {"convolution": {'weight':
                                 [
                                     'CODE',
                                     "np.reshape("\
                                     "self.tensor_to_numpy(tensor['C_1:out0']),"\
                                     " [self.shape_pick(tensor['C_1:out0'])[0], "\
                                     "self.shape_pick(tensor['C_1:out0'])[1], 1, -1])"
                                 ],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dwconv_quant_w_quant_o)

r_dwconv_add_quant_w = {
"ruler_name": "dwconv_add_quant_w",
"src_ops_alias": ["Add", "DepthwiseConv2dNative", "C", "FakeQuantWithMinMaxVars", "C_1", "C_2", "C_3"],
"src_inter_flow": [["C:out0", "Add:in1"], ["C_3:out0", "FakeQuantWithMinMaxVars:in2"],
                   ["C_1:out0", "FakeQuantWithMinMaxVars:in0"],
                   ["FakeQuantWithMinMaxVars:out0", "DepthwiseConv2dNative:in1"],
                   ["DepthwiseConv2dNative:out0", "Add:in0"], ["C_2:out0", "FakeQuantWithMinMaxVars:in1"]],
"src_in_anchor": [["I:out0", "DepthwiseConv2dNative:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': [
                                'INT',
                                'CODE',
                                "self.shape_pick(tensor['C_1:out0'])[2] * self.shape_pick(tensor['C_1:out0'])[3]"
                            ],
                            'group_number': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[2]"]}},
"blob_map": {"convolution": {'weight':
                                 [
                                     'CODE',
                                     "np.reshape("\
                                     "self.tensor_to_numpy(tensor['C_1:out0']),"\
                                     " [self.shape_pick(tensor['C_1:out0'])[0], "\
                                     "self.shape_pick(tensor['C_1:out0'])[1], 1, -1])"
                                 ],
                             'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dwconv_add_quant_w)

r_dwconv_badd = {
"ruler_name": "dw_bias",
"src_ops_alias": ["BiasAdd", "DepthwiseConv2dNative", "C", "C_1"],
"src_inter_flow": [["DepthwiseConv2dNative:out0", "BiasAdd:in0"], ["C:out0", "BiasAdd:in1"],
                   ["C_1:out0", "DepthwiseConv2dNative:in1"]],
"src_in_anchor": [["I:out0", "DepthwiseConv2dNative:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[1]"],
                            'stride_w': ['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights':
                                  [
                                      'INT',
                                      'CODE',
                                      "self.shape_pick(tensor['C_1:out0'])[2] * self.shape_pick(tensor['C_1:out0'])[3]"
                                  ],
                            'group_number': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[2]"]}},
"blob_map": {"convolution":
                 {
                     'weight':
                         [
                             'CODE',
                             "np.reshape("\
                             "self.tensor_to_numpy(tensor['C_1:out0']), "\
                             "[self.shape_pick(tensor['C_1:out0'])[0], self.shape_pick(tensor['C_1:out0'])[1], 1, -1])"
                         ],
                    'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]
                 }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dwconv_badd)

@rule_pyfunc_def
def r_dilation_conv_pad(self, node, tensor, pad_name, block_shape_name, crop_name):
    pads = self.tensor_to_numpy(tensor[pad_name])
    block_shapes = self.tensor_to_numpy(tensor[block_shape_name])
    crops = self.tensor_to_numpy(tensor[crop_name])
    conv_pads = [[0,0],[0,0]]
    for i in range(len(block_shapes)):
        conv_pads[i][0] = pads[i][0] - crops[i][0]
        conv_pads[i][1] = pads[i][1] - crops[i][1]
    import numpy as np
    conv_pads = np.asarray(conv_pads, np.int32)
    return conv_pads.flatten()

@rule_pyfunc_def
def r_dilation_conv_dilation(self, node, tensor, conv_name, block_shape_name):
    #Not support dilation in the batch and depth dim
    #TODO:maybe need add pre_condition
    conv_d = self.attr_pick(node[conv_name], 'dilations')[1]
    block_shape = self.tensor_to_numpy(tensor[block_shape_name])
    if len(block_shape) == 1:
        return [1, block_shape, block_shape, 1]  # nhwc
    elif len(block_shape) == 2:
        return [1, block_shape[0], block_shape[1], 1]  #nhwc
    else:
        raise ValueError('Unsupported block shape')

    return [1, 1, 1, 1]

@rule_pyfunc_def
def r_dilation_conv_pre_condition(self, node, tensor, kernel_name, conv_name, block_shape_name):
    kernel_h = self.shape_pick(tensor[kernel_name])[0]
    kernel_w = self.shape_pick(tensor[kernel_name])[1]
    stride_h = self.attr_pick(node[conv_name], 'strides')[1]
    stride_w = self.attr_pick(node[conv_name], 'strides')[2]
    dilation = self.attr_pick(node[conv_name], 'dilations')[1]
    block_shape = self.tensor_to_numpy(tensor[block_shape_name])[0]
    dilation = dilation * block_shape

    ker_h = (kernel_h - 1) * (dilation - 1) + kernel_h
    ker_w = (kernel_w - 1) * (dilation - 1) + kernel_w

    return (ker_h / stride_h < 16) and (ker_w / stride_w < 16)

r_dilation_conv_bias = {
"ruler_name": "dilation_conv_bias",
"src_ops_alias": ["BiasAdd", "BatchToSpaceND", "C", "Conv", "C_1", "C_2", "SpaceToBatchND", "C_3", "C_4", "C_5"],
"src_inter_flow": [["C_5:out0", "SpaceToBatchND:in2"], ["Conv:out0", "BatchToSpaceND:in0"],
                   ["C_2:out0", "BatchToSpaceND:in2"], ["C:out0", "BiasAdd:in1"], ["SpaceToBatchND:out0", "Conv:in0"],
                   ["C_1:out0", "BatchToSpaceND:in1"], ["C_4:out0", "SpaceToBatchND:in1"], ["C_3:out0", "Conv:in1"],
                   ["BatchToSpaceND:out0", "BiasAdd:in0"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding', 'SAME')"],
                            'pad_method': ['STRING', 'VALUE', "padding_const"],
                            'pad': ['INTS', 'PYFUNC', r_dilation_conv_pad(pad_name='C_5:out0',
                                block_shape_name='C_4:out0', crop_name='C_2:out0')],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[3]"],
                            'dilation': ['INTS', 'PYFUNC', r_dilation_conv_dilation(conv_name='Conv',
                                block_shape_name='C_4:out0')]}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_dilation_conv_pre_condition(kernel_name='C_3:out0', conv_name='Conv', block_shape_name='C_4:out0')}
ruler_list.append(r_dilation_conv_bias)

r_pad_dialated_conv_bias = {
"ruler_name": "pad_dialated_conv_bias",
"src_ops_alias": ["Pad", "C_6",
                  "BiasAdd", "BatchToSpaceND", "C", "Conv", "C_1", "C_2", "SpaceToBatchND", "C_3", "C_4", "C_5"],
"src_inter_flow": [["Pad:out0", "SpaceToBatchND:in0"],["C_6:out0", "Pad:in1"],
                   ["C_5:out0", "SpaceToBatchND:in2"], ["Conv:out0", "BatchToSpaceND:in0"],
                   ["C_2:out0", "BatchToSpaceND:in2"], ["C:out0", "BiasAdd:in1"], ["SpaceToBatchND:out0", "Conv:in0"],
                   ["C_1:out0", "BatchToSpaceND:in1"], ["C_4:out0", "SpaceToBatchND:in1"], ["C_3:out0", "Conv:in1"],
                   ["BatchToSpaceND:out0", "BiasAdd:in0"]],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding', 'SAME')"],
                            'pad_method': ['STRING', 'VALUE', "padding_const"],
                            'pad': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_6:out0'])[1:3].flatten()"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[3]"],
                            'dilation': ['INTS', 'PYFUNC', r_dilation_conv_dilation(conv_name='Conv',
                                block_shape_name='C_4:out0')]}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_dilation_conv_pre_condition(kernel_name='C_3:out0', conv_name='Conv', block_shape_name='C_4:out0')}
ruler_list.append(r_pad_dialated_conv_bias)

r_dialated_conv = {
"ruler_name": "dialated_conv",
"src_ops_alias": ["BatchToSpaceND", "BiasAdd", "C", "C_1", "Conv", "C_2", "SpaceToBatchND", "C_3", "C_4", "C_5"],
"src_inter_flow": [["C_3:out0", "Conv:in1"], ["Conv:out0", "BiasAdd:in0"], ["C:out0", "BatchToSpaceND:in1"],
                   ["BiasAdd:out0", "BatchToSpaceND:in0"], ["SpaceToBatchND:out0", "Conv:in0"],
                   ["C_2:out0", "BiasAdd:in1"], ["C_4:out0", "SpaceToBatchND:in1"], ["C_1:out0", "BatchToSpaceND:in2"],
                   ["C_5:out0", "SpaceToBatchND:in2"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["BatchToSpaceND:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BatchToSpaceND:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'VALUE', "SAME"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[3]"],
                            'dilation': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0'])[0]"]}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_dilation_conv_pre_condition(kernel_name='C_3:out0', conv_name='Conv', block_shape_name='C_4:out0')}
ruler_list.append(r_dialated_conv)

r_dilated_conv_no_biasadd = {
"ruler_name": "dilated_conv_no_biasadd",
"src_ops_alias": ["BatchToSpaceND", "Conv", "C", "C_1", "SpaceToBatchND", "C_2", "C_3", "C_4"],
"src_inter_flow": [["Conv:out0", "BatchToSpaceND:in0"], ["C:out0", "BatchToSpaceND:in1"],
                   ["C_1:out0", "BatchToSpaceND:in2"], ["SpaceToBatchND:out0", "Conv:in0"],
                   ["C_2:out0", "Conv:in1"], ["C_3:out0", "SpaceToBatchND:in1"], ["C_4:out0", "SpaceToBatchND:in2"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["BatchToSpaceND:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BatchToSpaceND:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding', 'SAME')"],
                            'pad_method': ['STRING', 'VALUE', "padding_const"],
                            'pad': ['INTS', 'PYFUNC', r_dilation_conv_pad(pad_name='C_4:out0',
                                block_shape_name='C_3:out0', crop_name='C_1:out0')],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[3]"],
                            'dilation': ['INTS', 'PYFUNC', r_dilation_conv_dilation(conv_name='Conv',
                                block_shape_name='C_3:out0')]}},
"blob_map": {"convolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"]}},
"priority_tip": 0,
"pre_condition": r_dilation_conv_pre_condition(kernel_name='C_2:out0', conv_name='Conv', block_shape_name='C_3:out0')}
ruler_list.append(r_dilated_conv_no_biasadd)

r_dialated_conv1d = {
"ruler_name": "dialated_conv1d",
"src_ops_alias": ["BatchToSpaceND", "Squeeze", "C", "C_1", "Conv", "ExpandDims",
                  "C_2", "SpaceToBatchND", "C_3", "C_4", "C_5"],
"src_inter_flow": [["Squeeze:out0", "BatchToSpaceND:in0"], ["C:out0", "BatchToSpaceND:in1"],
                   ["C_1:out0", "BatchToSpaceND:in2"], ["Conv:out0", "Squeeze:in0"],
                   ["ExpandDims:out0", "Conv:in0"], ["C_2:out0", "Conv:in1"],
                   ["SpaceToBatchND:out0", "ExpandDims:in0"], ["C_3:out0", "ExpandDims:in1"],
                   ["C_4:out0", "SpaceToBatchND:in1"], ["C_5:out0", "SpaceToBatchND:in2"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["BatchToSpaceND:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["BatchToSpaceND:out0", "conv1d:out0"]],
"acu_inter_flow": [],
"param_map": {"conv1d": {'ksize':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[1]"],
                        'stride':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                        'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                        'bias': ['BOOL', 'VALUE', False],
                        'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[3]"],
                        'dilation': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0'])[0]"]}},
"blob_map": {"conv1d": {'weight': ['CODE', "np.squeeze(self.tensor_to_numpy(tensor['C_2:out0']), 0)"]}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dialated_conv1d)

r_dialated_conv1d_bias = {
"ruler_name": "dialated_conv1d_bias",
"src_ops_alias": ["BiasAdd", "BatchToSpaceND", "C", "Squeeze", "C_1", "C_2", "Conv",
                  "ExpandDims", "C_3", "SpaceToBatchND", "C_4", "C_5", "C_6"],
"src_inter_flow": [["BatchToSpaceND:out0", "BiasAdd:in0"], ["C:out0", "BiasAdd:in1"],
                   ["Squeeze:out0", "BatchToSpaceND:in0"], ["C_1:out0", "BatchToSpaceND:in1"],
                   ["C_2:out0", "BatchToSpaceND:in2"], ["Conv:out0", "Squeeze:in0"],
                   ["ExpandDims:out0", "Conv:in0"], ["C_3:out0", "Conv:in1"],
                   ["SpaceToBatchND:out0", "ExpandDims:in0"], ["C_4:out0", "ExpandDims:in1"],
                   ["C_5:out0", "SpaceToBatchND:in1"], ["C_6:out0", "SpaceToBatchND:in2"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "conv1d:out0"]],
"acu_inter_flow": [],
"param_map": {"conv1d": {'ksize':['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[1]"],
                        'stride':['INT', 'CODE', "self.attr_pick(node['Conv'], 'strides')[2]"],
                        'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv'], 'padding')"],
                        'bias': ['BOOL', 'VALUE', True],
                        'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_3:out0'])[3]"],
                        'dilation': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_5:out0'])[0]"]}},
"blob_map": {"conv1d": {'weight': ['CODE', "np.squeeze(self.tensor_to_numpy(tensor['C_3:out0']), 0)"],
                        'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dialated_conv1d_bias)

# r_dialated_dwconv_bias = dict()
# r_dialated_dwconv_bias['src_ops_alias'] = ['SpaceToBatchND', 'DepthwiseConv2dNative', 'BatchToSpaceND', 'BiasAdd']
# r_dialated_dwconv_bias['acu_lys_alias'] = ['convolution']
# r_dialated_dwconv_bias['param_map'] = {'convolution':
# {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[0]"],
# 'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[1]"],
# 'stride_h':['INT', 'CODE', "self.attr_pick(tensor['DepthwiseConv2dNative'], 'strides')[1]"],
# 'stride_w':['INT', 'CODE', "self.attr_pick(tensor['DepthwiseConv2dNative'], 'strides')[2]"],
# 'padding': ['STRING', 'VALUE', "SAME"],
# 'bias': ['BOOL', 'VALUE', True],
# 'weights': ['INT', 'CODE',
# "self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[3] *
# self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[2]"],
# 'dilation': ['INT', 'CODE', "self.tensor_to_numpy(tensor['SpaceToBatchND'].input[1])[0]"],
# 'group_number': ['INT', 'CODE', "self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[2]"]
# }
#                     }
# r_dialated_dwconv_bias['blob_map'] = {'convolution':
#                           {'weight': ['CODE',
# "np.reshape( " \
#           "self.tensor_to_numpy(tensor['DepthwiseConv2dNative'].input[1]), "\
#           "[self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[0],"\
#           "self.shape_pick(tensor['DepthwiseConv2dNative'].input[1])[1], 1, -1])"],
#                            'bias': ['CODE', "self.tensor_to_numpy(tensor['BiasAdd'].input[1])"]
#                            }
#                    }
# ruler_list.append(r_dialated_dwconv_bias)
#
r_dialated_dwconv = {
"ruler_name": "dialated_deptwise",
"src_ops_alias": ["BatchToSpaceND", "DepthwiseConv2dNative", "C", "C_1", "SpaceToBatchND", "C_2", "C_3", "C_4"],
"src_inter_flow": [["DepthwiseConv2dNative:out0", "BatchToSpaceND:in0"], ["C_4:out0", "SpaceToBatchND:in2"],
                   ["C_1:out0", "BatchToSpaceND:in2"], ["C_2:out0", "DepthwiseConv2dNative:in1"],
                   ["C_3:out0", "SpaceToBatchND:in1"], ["C:out0", "BatchToSpaceND:in1"],
                   ["SpaceToBatchND:out0", "DepthwiseConv2dNative:in0"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["BatchToSpaceND:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["BatchToSpaceND:out0", "convolution:out0"]],
"param_map": {"convolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['DepthwiseConv2dNative'], 'strides')[2]"],
                            'padding': ['STRING', 'VALUE', "SAME"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights':
                                  [
                                      'INT',
                                      'CODE',
                                      "self.shape_pick(tensor['C_2:out0'])[3] * self.shape_pick(tensor['C_2:out0'])[2]"
                                  ],
                            'dilation': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])[0]"],
                            'group_number': ['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[2]"]}},
"blob_map": {"convolution": {'weight': ['CODE', "np.reshape( " \
                                              "self.tensor_to_numpy(tensor['C_2:out0']), "\
                                              "[self.shape_pick(tensor['C_2:out0'])[0],"\
                                              "self.shape_pick(tensor['C_2:out0'])[1], 1, -1])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dialated_dwconv)

r_dconvolution = {
"ruler_name": "single_deconvolution",
"src_ops_alias": ["DConv", "C", "C_1"],
"src_inter_flow": [["C_1:out0", "DConv:in1"], ["C:out0", "DConv:in0"]],
"src_in_anchor": [["I:out0", "DConv:in2"]],
"src_out_tensor": ["DConv:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["DConv:out0", "deconvolution:out0"]],
"param_map": {"deconvolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['DConv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['DConv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['DConv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[2]"],
                            'output_shape': ['INTS', 'CODE',
                                             "self.deconv_output_shape(self.tensor_to_numpy(tensor['C:out0']))"]}},
"blob_map": {"deconvolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dconvolution)

r_dconvolution_biasadd = {
"ruler_name": "dconvolution_biasadd",
"src_ops_alias": ["BiasAdd", "DConv", "C", "C_1", "C_2"],
"src_inter_flow": [["C:out0", "BiasAdd:in1"], ["C_2:out0", "DConv:in1"], ["DConv:out0", "BiasAdd:in0"],
                   ["C_1:out0", "DConv:in0"]],
"src_in_anchor": [["I:out0", "DConv:in2"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "deconvolution:out0"]],
"param_map": {"deconvolution": {'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[0]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['DConv'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['DConv'], 'strides')[2]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['DConv'], 'padding')"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[2]"],
                            'output_shape': ['INTS', 'CODE',
                                             "self.deconv_output_shape(self.tensor_to_numpy(tensor['C_1:out0']))"]}},
"blob_map": {"deconvolution": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dconvolution_biasadd)

r_conv3d_no_bias = {
"ruler_name": "conv3d_no_bias",
"src_ops_alias": ["Conv3D", "C"],
"src_inter_flow": [["C:out0", "Conv3D:in1"]],
"src_in_anchor": [["I:out0", "Conv3D:in0"]],
"src_out_tensor": ["Conv3D:out0"],
"acu_lys_alias": ["conv3d"],
"src_acu_in_tensor_map": [["I:out0", "conv3d:in0"]],
"src_acu_out_tensor_map": [["Conv3D:out0", "conv3d:out0"]],
"acu_inter_flow": [],
"param_map": {"conv3d": {'ksize_d': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
                            'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[1]"],
                            'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[2]"],
                            'stride_d':['INT', 'CODE', "self.attr_pick(node['Conv3D'], 'strides')[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv3D'], 'strides')[2]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv3D'], 'strides')[3]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv3D'], 'padding')"],
                            'pad_method': ['STRING', 'VALUE', "auto"],
                            'bias': ['BOOL', 'VALUE', False],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"]}},
"blob_map": {"conv3d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv3d_no_bias)

r_conv3d_with_bias = {
"ruler_name": "conv3d_with_bias",
"src_ops_alias": ["BiasAdd", "Conv3D", "C", "C_1"],
"src_inter_flow": [["Conv3D:out0", "BiasAdd:in0"], ["C:out0", "BiasAdd:in1"], ["C_1:out0", "Conv3D:in1"]],
"src_in_anchor": [["I:out0", "Conv3D:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["conv3d"],
"src_acu_in_tensor_map": [["I:out0", "conv3d:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "conv3d:out0"]],
"acu_inter_flow": [],
"param_map": {"conv3d": {'ksize_d':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[0]"],
                            'ksize_h':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
                            'ksize_w':['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[2]"],
                            'stride_d':['INT', 'CODE', "self.attr_pick(node['Conv3D'], 'strides')[1]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['Conv3D'], 'strides')[2]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['Conv3D'], 'strides')[3]"],
                            'padding': ['STRING', 'CODE', "self.attr_pick(node['Conv3D'], 'padding')"],
                            'pad_method': ['STRING', 'VALUE', "auto"],
                            'bias': ['BOOL', 'VALUE', True],
                            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[-1]"],}},
"blob_map": {"conv3d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_conv3d_with_bias)


r_lrn = {
"ruler_name": "lrn",
"src_ops_alias": ["LRN"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LRN:in0"]],
"src_out_tensor": ["LRN:out0"],
"acu_lys_alias": ["localresponsenormalization_tf"],
"src_acu_in_tensor_map": [["I:out0", "localresponsenormalization_tf:in0"]],
"src_acu_out_tensor_map": [["LRN:out0", "localresponsenormalization_tf:out0"]],
"param_map": {"localresponsenormalization_tf":
                  {'local_size':['INT', 'CODE', "self.attr_pick(node['LRN'], 'depth_radius') * 2 + 1"],
                    'bias':['FLOAT', 'CODE', "self.attr_pick(node['LRN'], 'bias', 1.0)"],
                    'alpha':[
                        'FLOAT',
                        'CODE',
                        "self.attr_pick("\
                        "node['LRN'], 'alpha', 1e-4)"
                    ],
                    'beta':['FLOAT', 'CODE', "self.attr_pick(node['LRN'], 'beta', 0.75)"],
                    'type':['STRING', 'VALUE', "NORM_ACROSS_CHANNELS"],}},
"blob_map": {"localresponsenormalization_tf": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_lrn)

r_l2normalize = {
"ruler_name": "l2normalize",
"src_ops_alias": ["Mul", "Rsqrt", "Maximum", "Sum",  "C", "Square", "C_1"],
"src_inter_flow": [["Rsqrt:out0", "Mul:in1"], ["Maximum:out0", "Rsqrt:in0"],
                   ["Sum:out0", "Maximum:in0"], ["C:out0", "Maximum:in1"],
                   ["Square:out0", "Sum:in0"], ["C_1:out0", "Sum:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Square:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["l2normalize"],
"src_acu_in_tensor_map": [["I:out0", "l2normalize:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "l2normalize:out0"]],
"acu_inter_flow": [],
"param_map": {"l2normalize":
                  {'l2n_dim': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                   'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_l2normalize)

r_l2normalize_l = {  # for lower version tf
    "ruler_name": "l2normalize_l",
    "src_ops_alias": ["RealDiv", "Sqrt", "Add", "C", "Sum",  "C_1", "Pow", "C_2"],
    "src_inter_flow": [["Sqrt:out0", "RealDiv:in1"], ["Add:out0", "Sqrt:in0"],
                       ["Sum:out0", "Add:in0"], ["C:out0", "Add:in1"],
                       ["Pow:out0", "Sum:in0"], ["C_1:out0", "Sum:in1"],
                       ["C_2:out0", "Pow:in1"]],
    "src_in_anchor": [["I:out0", "RealDiv:in0"], ["I:out0", "Pow:in0"]],
    "src_out_tensor": ["RealDiv:out0"],
    "acu_lys_alias": ["l2normalize"],
    "src_acu_in_tensor_map": [["I:out0", "l2normalize:in0"]],
    "src_acu_out_tensor_map": [["RealDiv:out0", "l2normalize:out0"]],
    "acu_inter_flow": [],
    "param_map": {"l2normalize":
                      {'l2n_dim': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                       'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
    "blob_map": {},
    "priority_tip": 0,
    "pre_condition": "self.tensor_to_numpy(tensor['C_2:out0']) == 2"}
ruler_list.append(r_l2normalize_l)

r_l2normscale = {
"ruler_name": "l2normscale",
"src_ops_alias": ["Mul", "Mul_1", "C", "Rsqrt", "Maximum", "Sum", "C_1", "Square", "C_2"],
"src_inter_flow": [["Mul_1:out0", "Mul:in0"], ["C:out0", "Mul:in1"],
                   ["Rsqrt:out0", "Mul_1:in1"], ["Maximum:out0", "Rsqrt:in0"],
                   ["Sum:out0", "Maximum:in0"], ["C_1:out0", "Maximum:in1"],
                   ["Square:out0", "Sum:in0"], ["C_2:out0", "Sum:in1"]],
"src_in_anchor": [["I:out0", "Mul_1:in0"], ["I:out0", "Square:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["l2normalizescale"],
"src_acu_in_tensor_map": [["I:out0", "l2normalizescale:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "l2normalizescale:out0"]],
"acu_inter_flow": [],
"param_map": {"l2normalizescale":
                  {'l2n_dim': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                   'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"]}},
"blob_map": {"l2normalizescale":
                 {'scale': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_l2normscale)

r_bn = {
"ruler_name": "fusebatchnorm",
"src_ops_alias": ["FusedBatchNorm", "C", "C_1", "C_2", "C_3"],
"src_inter_flow": [["C_3:out0", "FusedBatchNorm:in4"], ["C_2:out0", "FusedBatchNorm:in3"],
                   ["C:out0", "FusedBatchNorm:in1"], ["C_1:out0", "FusedBatchNorm:in2"]],
"src_in_anchor": [["I:out0", "FusedBatchNorm:in0"]],
"src_out_tensor": ["FusedBatchNorm:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["FusedBatchNorm:out0", "batchnormalize:out0"]],
"param_map": {"batchnormalize": {'eps': ['FLOAT', 'CODE', "self.attr_pick(node['FusedBatchNorm'], 'epsilon', 1e-4)"]}},
"blob_map":
    {"batchnormalize":
        {
           'gamma':
               [
                   'CODE',
                   "None "\
                   "if len(self.tensor_to_numpy(tensor['C:out0'])) == 0 "\
                   "else self.tensor_to_numpy(tensor['C:out0'])"
               ],
           'beta':
               [
                   'CODE',
                   "None "\
                   "if len(self.tensor_to_numpy(tensor['C_1:out0'])) == 0 "\
                   "else self.tensor_to_numpy(tensor['C_1:out0'])"
               ],
           'mean':
               [
                   'CODE',
                   "None "\
                   "if len(self.tensor_to_numpy(tensor['C_2:out0'])) == 0 "\
                   "else self.tensor_to_numpy(tensor['C_2:out0'])"
               ],
           'variance':
               [
                   'CODE',
                   "None "\
                   "if len(self.tensor_to_numpy(tensor['C_3:out0'])) == 0 "\
                   "else self.tensor_to_numpy(tensor['C_3:out0'])"
               ],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_bn)

r_bnv3 = {
    "ruler_name": "fusebatchnormv3",
    "src_ops_alias": ["FusedBatchNormV3", "C", "C_1", "C_2", "C_3"],
    "src_inter_flow": [["C_3:out0", "FusedBatchNormV3:in4"], ["C_2:out0", "FusedBatchNormV3:in3"],
                       ["C:out0", "FusedBatchNormV3:in1"], ["C_1:out0", "FusedBatchNormV3:in2"]],
    "src_in_anchor": [["I:out0", "FusedBatchNormV3:in0"]],
    "src_out_tensor": ["FusedBatchNormV3:out0"],
    "acu_lys_alias": ["batchnormalize"],
    "src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
    "src_acu_out_tensor_map": [["FusedBatchNormV3:out0", "batchnormalize:out0"]],
    "param_map": {"batchnormalize": {'eps':['FLOAT', 'CODE', "self.attr_pick(node['FusedBatchNormV3'],"\
                                                             "'epsilon', 1e-4)"]}},
    "blob_map":
        {"batchnormalize":
            {
                'gamma':
                    [
                        'CODE',
                        "None " \
                        "if len(self.tensor_to_numpy(tensor['C:out0'])) == 0 " \
                        "else self.tensor_to_numpy(tensor['C:out0'])"
                    ],
                'beta':
                    [
                        'CODE',
                        "None " \
                        "if len(self.tensor_to_numpy(tensor['C_1:out0'])) == 0 " \
                        "else self.tensor_to_numpy(tensor['C_1:out0'])"
                    ],
                'mean':
                    [
                        'CODE',
                        "None " \
                        "if len(self.tensor_to_numpy(tensor['C_2:out0'])) == 0 " \
                        "else self.tensor_to_numpy(tensor['C_2:out0'])"
                    ],
                'variance':
                    [
                        'CODE',
                        "None " \
                        "if len(self.tensor_to_numpy(tensor['C_3:out0'])) == 0 " \
                        "else self.tensor_to_numpy(tensor['C_3:out0'])"
                    ],
            }},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_bnv3)

r_bn_as_instancenorm = {
"ruler_name": "fusebatchnorm_as_instancenorm",
"src_ops_alias": ["FusedBatchNorm", "C", "C_1", "C_2", "C_3"],
"src_inter_flow": [["C_3:out0", "FusedBatchNorm:in4"], ["C_2:out0", "FusedBatchNorm:in3"],
                   ["C:out0", "FusedBatchNorm:in1"], ["C_1:out0", "FusedBatchNorm:in2"]],
"src_in_anchor": [["I:out0", "FusedBatchNorm:in0"]],
"src_out_tensor": ["FusedBatchNorm:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["FusedBatchNorm:out0", "instancenormalize:out0"]],
"param_map": {
    "instancenormalize": {'eps': ['FLOAT', 'CODE', "self.attr_pick(node['FusedBatchNorm'], 'epsilon', 1e-4)"],
                            'axis': ['INTS', 'VALUE', [0, 1, 2]]}},
"blob_map": {"instancenormalize":
                                {'bias': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                }},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": "self.attr_pick(node['FusedBatchNorm'], 'is_training', True) == True "\
                 "and len(self.tensor_to_numpy(tensor['C_2:out0'])) == 0 "\
                 "and len(self.tensor_to_numpy(tensor['C_3:out0'])) == 0"}
ruler_list.append(r_bn_as_instancenorm)

r_mul_add_2_bn ={
"ruler_name": "mul_add_2_bn",
"src_ops_alias": ["Add", "Mul", "C", "C_1"],
"src_inter_flow": [["C:out0", "Add:in1"], ["Mul:out0", "Add:in0"], ["C_1:out0", "Mul:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnormalize:out0"]],
"param_map": {"batchnormalize": {'eps': ['FLOAT', 'VALUE', 1e-5]}},
"blob_map": {"batchnormalize": {'beta': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                           'gamma': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['C:out0'])) == 1 "\
                 "and self.shape_pick(tensor['C:out0'])[0] == self.shape_pick(tensor['I:out0'])[-1]" \
                 "and len(self.shape_pick(tensor['C_1:out0'])) == 1 "\
                 "and self.shape_pick(tensor['C_1:out0'])[0] == self.shape_pick(tensor['C:out0'])[0]"}
ruler_list.append(r_mul_add_2_bn)

@rule_pyfunc_def
def r_instancenorm_pre_condition(self, node, tensor, input_names):
    input_shape = self.shape_pick(tensor[input_names[0]])
    axis = self.tensor_to_numpy(tensor[input_names[1]]).tolist()
    # TODO:maybe need refine the condition
    #return ((len(input_shape) == 4 or len(input_shape) == 3) and axis == [1,2])\
    #       or (len(input_shape) == 2 and axis == [0,1])
    return (len(input_shape) == 4 and axis == [1,2]) or \
           (len(input_shape) == 4 and axis == [0,1,2] and input_shape[0] == 1)

#tf.contrib.layers.instance_norm scale=True center=True
r_instancenorm_tf_1_13_scale_true_center_true = {
"ruler_name": "r_instancenorm_tf_1_13_scale_true_center_true",
"src_ops_alias": ["Add", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Mean", "Add_1", "C_2", "Mean_1", "C_3",
                  "SquaredDifference", "C_4", "StopGradient"],
"src_inter_flow":
    [
        ["C_1:out0", "Mul_1:in1"], ["StopGradient:out0", "SquaredDifference:in1"], ["Mul:out0", "Add:in0"],
        ["C_4:out0", "Mean_1:in1"], ["Mean:out0", "StopGradient:in0"],["Mean:out0", "Mul_2:in0"],
        ["Rsqrt:out0", "Mul_1:in0"], ["Add_1:out0", "Rsqrt:in0"], ["Mul_1:out0", "Mul_2:in1"],["C:out0", "Sub:in0"],
        ["C_2:out0", "Mean:in1"],["Mul_2:out0", "Sub:in1"], ["SquaredDifference:out0", "Mean_1:in0"],
        ["Mul_1:out0", "Mul:in1"], ["C_3:out0", "Add_1:in1"], ["Mean_1:out0", "Add_1:in0"], ["Sub:out0", "Add:in1"]
    ],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'VALUE', 1e-5],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0']).tolist()"]}},
"blob_map": {"instancenormalize":
                                {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                                }},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_4:out0'])}
ruler_list.append(r_instancenorm_tf_1_13_scale_true_center_true)


r_instancenorm_1 = {
"ruler_name": "instance_normalize",
"src_ops_alias": ["Add", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Add_1", "Add_2", "Mul_3", "Reshape",
                  "Sub_1", "C_2", "Sum", "C_3", "StopGradient", "C_4", "Mul_4",
    "Square", "Sub_2", "C_5", "Mean", "Sum_1", "C_6", "SquaredDifference", "C_7"],
"src_inter_flow": [["Sum_1:out0", "Mul_4:in0"], ["Mul_2:out0", "Sub:in1"], ["Sub_1:out0", "Add_2:in0"],
                   ["Sum:out0", "Mul_3:in0"], ["StopGradient:out0", "SquaredDifference:in1"],
    ["Add_2:out0", "Rsqrt:in0"], ["Sub:out0", "Add:in1"], ["C_4:out0", "Reshape:in1"],
                   ["C_6:out0", "Mean:in1"], ["Reshape:out0", "Add_1:in1"], ["Mul_4:out0", "Sub_1:in0"],
    ["Mul:out0", "Add:in0"], ["C_5:out0", "Sum:in1"], ["StopGradient:out0", "Sub_2:in1"],
                   ["SquaredDifference:out0", "Sum_1:in0"], ["C_3:out0", "Mul_4:in1"], ["Mul_3:out0", "Add_1:in0"],
    ["Mean:out0", "StopGradient:in0"], ["C_7:out0", "Sum_1:in1"], ["Mul_1:out0", "Mul:in1"],
                   ["C_1:out0", "Mul_1:in1"], ["Sub_2:out0", "Sum:in0"], ["C_2:out0", "Add_2:in1"],
    ["Mul_3:out0", "Square:in0"], ["Add_1:out0", "Mul_2:in0"], ["Square:out0", "Sub_1:in1"],
                   ["Mul_1:out0", "Mul_2:in1"], ["StopGradient:out0", "Reshape:in0"], ["Rsqrt:out0", "Mul_1:in0"],
    ["C_3:out0", "Mul_3:in1"], ["C:out0", "Sub:in0"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "Sub_2:in0"], ["I:out0", "SquaredDifference:in0"],
                  ["I:out0", "Mul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_6:out0']).tolist()"]}},
"blob_map": {"instancenormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_6:out0'])}
ruler_list.append(r_instancenorm_1)

r_instancenorm_2 = {
"ruler_name": "instance_norm_2",
"src_ops_alias": ["Add", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Squeeze", "Add_1", "Mean", "Squeeze_1",
                  "C_2", "C_3", "Mean_1", "SquaredDifference", "C_4", "StopGradient"],
"src_inter_flow": [["Sub:out0", "Add:in1"], ["Mean_1:out0", "Squeeze_1:in0"], ["Mul_1:out0", "Mul:in1"],
                   ["Mul:out0", "Add:in0"], ["Rsqrt:out0", "Mul_1:in0"], ["C_3:out0", "Mean:in1"],
    ["Add_1:out0", "Rsqrt:in0"], ["Mean:out0", "StopGradient:in0"], ["Squeeze_1:out0", "Add_1:in0"],
                   ["C_4:out0", "Mean_1:in1"], ["Squeeze:out0", "Mul_2:in0"], ["C_2:out0", "Add_1:in1"],
    ["Mul_2:out0", "Sub:in1"], ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "Squeeze:in0"],
                   ["SquaredDifference:out0", "Mean_1:in0"], ["Mul_1:out0", "Mul_2:in1"],
    ["C_1:out0", "Mul_1:in1"], ["C:out0", "Sub:in0"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"], ["I:out0", "Mul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0']).tolist()"]}},
"blob_map": {"instancenormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_3:out0'])}
ruler_list.append(r_instancenorm_2)

r_instancenorm_3 = {
"ruler_name": "instancenormalize",
"src_ops_alias": ["Add", "Mul", "C", "C_1", "RealDiv", "Sub", "Pow", "Mean", "Add_1", "C_2", "C_3", "Mean_1", "C_4",
                  "SquaredDifference", "C_5"],
"src_inter_flow": [["C_4:out0", "Add_1:in1"], ["Mean_1:out0", "Add_1:in0"], ["C_3:out0", "Mean:in1"],
                   ["C_1:out0", "Mul:in0"], ["Mul:out0", "Add:in0"], ["C:out0", "Add:in1"], ["Add_1:out0", "Pow:in0"],
    ["RealDiv:out0", "Mul:in1"], ["C_2:out0", "Pow:in1"], ["Sub:out0", "RealDiv:in0"],
                   ["Mean:out0", "SquaredDifference:in1"], ["Pow:out0", "RealDiv:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_5:out0", "Mean_1:in1"], ["Mean:out0", "Sub:in1"]],
"src_in_anchor": [["I:out0", "Sub:in0"], ["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0']).tolist()"]}},
"blob_map": {"instancenormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_3:out0'])}
ruler_list.append(r_instancenorm_3)

r_instancenorm_4 = {
"ruler_name": 'r_instancenorm_4',
"src_ops_alias": ["Add", "Mul", "C", "C_1", "RealDiv", "Sub", "Pow", "Mean", "Add_1", "C_2","C_3",
                  "Mean_1", "C_4", "SquaredDifference", "C_5", "StopGradient"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["C:out0", "Add:in1"], ["C_1:out0", "Mul:in0"],
                   ["RealDiv:out0", "Mul:in1"], ["Sub:out0", "RealDiv:in0"], ["Pow:out0", "RealDiv:in1"],
                   ["Mean:out0", "Sub:in1"], ["Add_1:out0", "Pow:in0"], ["C_2:out0", "Pow:in1"],
                   ["C_3:out0", "Mean:in1"], ["Mean_1:out0", "Add_1:in0"], ["C_4:out0", "Add_1:in1"],
                   ["SquaredDifference:out0", "Mean_1:in0"], ["C_5:out0", "Mean_1:in1"],
                   ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"], ["I:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0']).tolist()"]}},
"blob_map": {"instancenormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                                }},
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_3:out0'])}
ruler_list.append(r_instancenorm_4)

r_instancenorm_5 = {
"ruler_name": "r_instancenorm_5",
"src_ops_alias": ["Add", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Mean", "Add_1", "C_2", "Mean_1", "C_3",
    "SquaredDifference", "C_4"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Sub:out0", "Add:in1"], ["Mul_1:out0", "Mul:in1"], ["C:out0", "Sub:in0"],
    ["Mul_2:out0", "Sub:in1"], ["Rsqrt:out0", "Mul_1:in0"], ["C_1:out0", "Mul_1:in1"],
    ["Mul_1:out0", "Mul_2:in1"], ["Mean:out0", "Mul_2:in0"], ["Add_1:out0", "Rsqrt:in0"],
    ["C_2:out0", "Mean:in1"], ["Mean_1:out0", "Add_1:in0"], ["C_3:out0", "Add_1:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_4:out0", "Mean_1:in1"],
    ["Mean:out0", "SquaredDifference:in1"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "Mul:in0"], ["I:out0", "SquaredDifference:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0']).tolist()"]}},
"blob_map": {"instancenormalize":
                                {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                                }},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_4:out0'])}
ruler_list.append(r_instancenorm_5)

r_instancenorm_6 = {
"ruler_name": "r_instancenorm_6",
"src_ops_alias": ["AddV2", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Mean", "AddV2_1", "C_2", "Mean_1",
    "C_3", "SquaredDifference", "C_4", "StopGradient"],
"src_inter_flow": [["Mul:out0", "AddV2:in0"], ["Sub:out0", "AddV2:in1"], ["Mul_1:out0", "Mul:in1"],
    ["C:out0", "Sub:in0"],
    ["Mul_2:out0", "Sub:in1"], ["Rsqrt:out0", "Mul_1:in0"], ["C_1:out0", "Mul_1:in1"],
    ["Mul_1:out0", "Mul_2:in1"], ["Mean:out0", "Mul_2:in0"], ["AddV2_1:out0", "Rsqrt:in0"],
    ["C_2:out0", "Mean:in1"], ["Mean_1:out0", "AddV2_1:in0"], ["C_3:out0", "AddV2_1:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_4:out0", "Mean_1:in1"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"]],
"src_out_tensor": ["AddV2:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["AddV2:out0", "instancenormalize:out0"]],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0']).tolist()"]}},
"blob_map": {"instancenormalize":
                                {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                                }},
"acu_inter_flow": [],
"priority_tip": 0,
#"pre_condition": None}
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_4:out0'])}
ruler_list.append(r_instancenorm_6)


#tf.contrib.layers.instance_norm scale=False center=False
r_instancenorm_tf_1_13_scale_false_center_false = {
"ruler_name": "r_instancenorm_tf_1_13_scale_false_center_false",
"src_ops_alias": ["Add", "Mul", "Mul_1", "Rsqrt", "Neg", "Add_1", "Mean", "Mean_1", "C", "C_1", "SquaredDifference",
    "C_2", "StopGradient"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Mul_1:out0", "Add:in1"], ["Rsqrt:out0", "Mul:in1"],
    ["Rsqrt:out0", "Mul_1:in1"], ["Neg:out0", "Mul_1:in0"], ["Add_1:out0", "Rsqrt:in0"],
    ["Mean:out0", "Neg:in0"], ["Mean_1:out0", "Add_1:in0"], ["C:out0", "Add_1:in1"],
    ["C_1:out0", "Mean:in1"], ["SquaredDifference:out0", "Mean_1:in0"], ["C_2:out0", "Mean_1:in1"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0']).tolist()"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_1:out0'])}
ruler_list.append(r_instancenorm_tf_1_13_scale_false_center_false)

#tf.contrib.layers.instance_norm scale=True center=False
r_instancenorm_tf_1_13_scale_true_center_false = {
"ruler_name": "r_instancenorm_tf_1_13_scale_true_center_false",
"src_ops_alias": ["Add", "Mul", "Mul_1", "Mul_2", "Neg", "Rsqrt", "C", "Mean", "Add_1", "C_1", "Mean_1", "C_2",
    "SquaredDifference", "C_3", "StopGradient"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Mul_1:out0", "Add:in1"], ["Mul_2:out0", "Mul:in1"],
    ["Mul_2:out0", "Mul_1:in1"], ["Neg:out0", "Mul_1:in0"], ["Rsqrt:out0", "Mul_2:in0"],
    ["C:out0", "Mul_2:in1"], ["Mean:out0", "Neg:in0"], ["Add_1:out0", "Rsqrt:in0"],
    ["C_1:out0", "Mean:in1"], ["Mean_1:out0", "Add_1:in0"], ["C_2:out0", "Add_1:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_3:out0", "Mean_1:in1"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "SquaredDifference:in0"], ["I:out0", "Mul:in0"], ["I:out0", "Mean:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0']).tolist()"]}},
"blob_map": {"instancenormalize":
                                {
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                }},
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_1:out0'])}
ruler_list.append(r_instancenorm_tf_1_13_scale_true_center_false)

#tf.contrib.layers.instance_norm scale=False center=True
r_instancenorm_tf_1_13_scale_false_center_true = {
"ruler_name": "r_instancenorm_tf_1_13_scale_false_center_true",
"src_ops_alias": ["Add", "Mul", "Sub", "Rsqrt", "C", "Mul_1", "Add_1", "Mean", "Mean_1", "C_1", "C_2",
    "SquaredDifference", "C_3", "StopGradient"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Sub:out0", "Add:in1"], ["Rsqrt:out0", "Mul:in1"], ["C:out0", "Sub:in0"],
    ["Mul_1:out0", "Sub:in1"], ["Add_1:out0", "Rsqrt:in0"], ["Rsqrt:out0", "Mul_1:in1"],
    ["Mean:out0", "Mul_1:in0"], ["Mean_1:out0", "Add_1:in0"], ["C_1:out0", "Add_1:in1"],
    ["C_2:out0", "Mean:in1"], ["SquaredDifference:out0", "Mean_1:in0"], ["C_3:out0", "Mean_1:in1"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"], ["I:out0", "Mul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "instancenormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"instancenormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                            'axis': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0']).tolist()"]}},
"blob_map": {"instancenormalize":
                                {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                }},
"priority_tip": 0,
"pre_condition": r_instancenorm_pre_condition(input_names=['I:out0', 'C_2:out0'])}
ruler_list.append(r_instancenorm_tf_1_13_scale_false_center_true)

@rule_pyfunc_def
def r_layernorm_pre_condition(self, node, tensor, axis_tensor_name, input_tensor_name):
    ''' the axis list only does not contain the batch dim '''
    axis = self.tensor_to_numpy(tensor[axis_tensor_name])
    _shape = list(self.shape_pick(tensor[input_tensor_name]))
    input_dims = len(_shape)
    if axis.shape == 0:  # for scalar
        axis = [axis]
    need_axis = list(range(input_dims))
    need_axis.pop(0)  # no batch dim
    return list(axis) == need_axis

r_layernorm = {
"ruler_name": "layer_norm",
"src_ops_alias": ["Add", "Mul", "C", "Mul_1", "C_1", "Sub", "Rsqrt", "Mean", "Add_1", "C_2", "Mean_1", "C_3", "Square",
                  "C_4", "Sub_1"],
"src_inter_flow": [["Sub_1:out0", "Square:in0"], ["Sub:out0", "Mul_1:in0"], ["Mul:out0", "Add:in0"],
                   ["C:out0", "Add:in1"], ["C_3:out0", "Add_1:in1"], ["Add_1:out0", "Rsqrt:in0"],
    ["Mean:out0", "Sub:in1"], ["C_2:out0", "Mean:in1"], ["Mean_1:out0", "Add_1:in0"], ["Square:out0", "Mean_1:in0"],
                   ["Mean:out0", "Sub_1:in1"],
    ["Rsqrt:out0", "Mul_1:in1"], ["C_1:out0", "Mul:in1"], ["C_4:out0", "Mean_1:in1"], ["Mul_1:out0", "Mul:in0"]],
"src_in_anchor": [["I:out0", "Sub:in0"], ["I:out0", "Sub_1:in0"], ["I:out0", "Mean:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"param_map": {"layernormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],}},
"blob_map": {"layernormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_layernorm_pre_condition(axis_tensor_name='C_2:out0', input_tensor_name='I:out0')}
ruler_list.append(r_layernorm)

r_layernorm_sqrdiff = {
"ruler_name": "layer_norm_sqrdiff",
"src_ops_alias": ["Add", "Mul", "C", "Mul_1", "C_1", "Sub", "Rsqrt", "Mean", "Add_1", "C_2", "Mean_1", "C_3",
    "SquaredDifference", "C_4"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["C:out0", "Add:in1"], ["Mul_1:out0", "Mul:in0"], ["C_1:out0", "Mul:in1"],
    ["Sub:out0", "Mul_1:in0"], ["Rsqrt:out0", "Mul_1:in1"], ["Mean:out0", "Sub:in1"],
    ["Add_1:out0", "Rsqrt:in0"], ["C_2:out0", "Mean:in1"], ["Mean_1:out0", "Add_1:in0"],
    ["C_3:out0", "Add_1:in1"], ["SquaredDifference:out0", "Mean_1:in0"], ["C_4:out0", "Mean_1:in1"],
    ["Mean:out0", "SquaredDifference:in1"]],
"src_in_anchor": [["I:out0", "SquaredDifference:in0"], ["I:out0", "Mean:in0"], ["I:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"layernormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],}},
"blob_map": {"layernormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": r_layernorm_pre_condition(axis_tensor_name='C_2:out0', input_tensor_name='I:out0')}
ruler_list.append(r_layernorm_sqrdiff)

r_layernorm_sqrdiffv2 = {
"ruler_name": "layer_norm_sqrdiffv2",
"src_ops_alias": ["Add", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Mean", "Add_1", "C_2", "Mean_1", "C_3",
    "SquaredDifference", "C_4"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Sub:out0", "Add:in1"], ["Mul_1:out0", "Mul:in1"], ["C:out0", "Sub:in0"],
    ["Mul_2:out0", "Sub:in1"], ["Rsqrt:out0", "Mul_1:in0"], ["C_1:out0", "Mul_1:in1"],
    ["Mul_1:out0", "Mul_2:in1"], ["Mean:out0", "Mul_2:in0"], ["Add_1:out0", "Rsqrt:in0"],
    ["C_2:out0", "Mean:in1"], ["Mean_1:out0", "Add_1:in0"], ["C_3:out0", "Add_1:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_4:out0", "Mean_1:in1"],
    ["Mean:out0", "SquaredDifference:in1"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"], ["I:out0", "Mul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"layernormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],}},
"blob_map": {"layernormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_layernorm_sqrdiffv2)

r_layernorm_no_scale_bias = {
"ruler_name": "r_layernorm_no_scale_bias",
"src_ops_alias": ["Add", "Mul", "Mul_1", "Reshape", "Rsqrt", "Neg", "C", "Add_1", "Mean", "Mean_1", "C_1", "C_2",
    "SquaredDifference", "C_3", "StopGradient"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Mul_1:out0", "Add:in1"], ["Reshape:out0", "Mul:in0"],
    ["Rsqrt:out0", "Mul:in1"], ["Rsqrt:out0", "Mul_1:in1"], ["Neg:out0", "Mul_1:in0"],
    ["C:out0", "Reshape:in1"], ["Add_1:out0", "Rsqrt:in0"], ["Mean:out0", "Neg:in0"],
    ["Mean_1:out0", "Add_1:in0"], ["C_1:out0", "Add_1:in1"], ["Reshape:out0", "Mean:in0"],
    ["C_2:out0", "Mean:in1"], ["SquaredDifference:out0", "Mean_1:in0"], ["C_3:out0", "Mean_1:in1"],
    ["Reshape:out0", "SquaredDifference:in0"], ["StopGradient:out0", "SquaredDifference:in1"],
    ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["reshape", "layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [["reshape:out0", "layernormalize:in0"]],
"param_map": {
    "layernormalize": {
        'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
    },
    "reshape": {
        'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C:out0'])"],
    }
},
"blob_map": {
    "layernormalize": {}
},
"priority_tip": 0,
"pre_condition": None
}
ruler_list.append(r_layernorm_no_scale_bias)

r_layernorm_with_moments = {
"ruler_name": "layernorm_with_moment",
"src_ops_alias": ["AddV2", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Mean", "AddV2_1", "C_2", "Mean_1",
    "C_3", "SquaredDifference", "C_4", "StopGradient"],
"src_inter_flow": [["Mul:out0", "AddV2:in0"], ["Sub:out0", "AddV2:in1"],
    ["Mul_1:out0", "Mul:in1"], ["C:out0", "Sub:in0"],
    ["Mul_2:out0", "Sub:in1"], ["Rsqrt:out0", "Mul_1:in0"], ["C_1:out0", "Mul_1:in1"],
    ["Mul_1:out0", "Mul_2:in1"], ["Mean:out0", "Mul_2:in0"], ["AddV2_1:out0", "Rsqrt:in0"],
    ["C_2:out0", "Mean:in1"], ["Mean_1:out0", "AddV2_1:in0"], ["C_3:out0", "AddV2_1:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_4:out0", "Mean_1:in1"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"], ["I:out0", "Mul:in0"]],
"src_out_tensor": ["AddV2:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["AddV2:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"layernormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
                                 'axis_list': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0']).tolist()"],}},
"blob_map": {"layernormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": r_layernorm_pre_condition(axis_tensor_name='C_2:out0', input_tensor_name='I:out0')
}
ruler_list.append(r_layernorm_with_moments)

r_layernorm1 = {
"ruler_name": "r_layernormalize1",
"src_ops_alias": ["Add", "Mul", "C", "RealDiv", "C_1", "Sub", "Add_1", "Mean", "Sqrt", "C_2", "C_3",
                  "Mean_1", "Square", "C_4", "Sub_1", "Mean_2", "C_5"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["C:out0", "Add:in1"], ["RealDiv:out0", "Mul:in0"],
    ["C_1:out0", "Mul:in1"], ["Sub:out0", "RealDiv:in0"], ["Add_1:out0", "RealDiv:in1"], ["Mean:out0", "Sub:in1"],
    ["Sqrt:out0", "Add_1:in0"], ["C_2:out0", "Add_1:in1"], ["C_3:out0", "Mean:in1"],
    ["Mean_1:out0", "Sqrt:in0"], ["Square:out0", "Mean_1:in0"], ["C_4:out0", "Mean_1:in1"],
    ["Sub_1:out0", "Square:in0"], ["Mean_2:out0", "Sub_1:in1"], ["C_5:out0", "Mean_2:in1"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "Mean_2:in0"], ["I:out0", "Sub_1:in0"], ["I:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"layernormalize": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                                 'axis_list': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0']).tolist()"],}},
"blob_map": {"layernormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": r_layernorm_pre_condition(axis_tensor_name='C_3:out0', input_tensor_name='I:out0')
}
ruler_list.append(r_layernorm1)

r_moments = {
"ruler_name": "moments",
"src_ops_alias": ["Mean", "Mean_1", "C", "SquaredDifference", "C_1", "StopGradient"],
"src_inter_flow": [["C:out0", "Mean:in1"], ["SquaredDifference:out0", "Mean_1:in0"], ["C_1:out0", "Mean_1:in1"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Mean:in0"], ["I:out0", "SquaredDifference:in0"]],
"src_out_tensor": ["Mean:out0", "Mean_1:out0"],
"acu_lys_alias": ["moments"],
"src_acu_in_tensor_map": [["I:out0", "moments:in0"]],
"src_acu_out_tensor_map": [["Mean:out0", "moments:out0"], ["Mean_1:out0", "moments:out1"]],
"acu_inter_flow": [],
"param_map": {"moments": {'axis_list': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"],
                          'keep_dims': ['BOOL', 'VALUE', True]
                          }},
"blob_map": {"moments": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_moments)

r_moments2 = {
"ruler_name": "r_moments2",
"src_ops_alias": ["Sub", "Add", "Mul", "Square", "Mul_1", "Reshape", "Sum", "C", "Sum_1", "StopGradient", "C_1",
    "SquaredDifference", "C_2", "Sub_1", "C_3", "Mean", "C_4"],
"src_inter_flow": [["Mul:out0", "Sub:in0"], ["Square:out0", "Sub:in1"], ["Mul_1:out0", "Add:in0"],
    ["Reshape:out0", "Add:in1"], ["Sum:out0", "Mul:in0"], ["C:out0", "Mul:in1"],
    ["Mul_1:out0", "Square:in0"], ["C:out0", "Mul_1:in1"], ["Sum_1:out0", "Mul_1:in0"],
    ["StopGradient:out0", "Reshape:in0"], ["C_1:out0", "Reshape:in1"],
    ["SquaredDifference:out0", "Sum:in0"], ["C_2:out0", "Sum:in1"], ["Sub_1:out0", "Sum_1:in0"],
    ["C_3:out0", "Sum_1:in1"], ["Mean:out0", "StopGradient:in0"],
    ["StopGradient:out0", "SquaredDifference:in1"], ["StopGradient:out0", "Sub_1:in1"],
    ["C_4:out0", "Mean:in1"]],
"src_in_anchor": [["I:out0", "SquaredDifference:in0"], ["I:out0", "Mean:in0"], ["I:out0", "Sub_1:in0"]],
"src_out_tensor": ["Sub:out0", "Add:out0"],
"acu_lys_alias": ["moments"],
"src_acu_in_tensor_map": [["I:out0", "moments:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "moments:out0"], ["Sub:out0", "moments:out1"]],
"acu_inter_flow": [],
"param_map": {"moments": {'axis_list': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0']).tolist()"],
                          'keep_dims': ['BOOL', 'VALUE', True]
                          }},
"blob_map": {"moments": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_moments2)

r_batchnorm_single = {
"ruler_name": "batchnorm_single",
"src_ops_alias": ["Add", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "Add_1", "C_2"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Sub:out0", "Add:in1"], ["Mul_1:out0", "Mul:in1"], ["C:out0", "Sub:in0"],
    ["Mul_2:out0", "Sub:in1"], ["Rsqrt:out0", "Mul_1:in0"], ["C_1:out0", "Mul_1:in1"],
    ["Mul_1:out0", "Mul_2:in1"], ["Add_1:out0", "Rsqrt:in0"], ["C_2:out0", "Add_1:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I_1:out0", "Mul_2:in0"], ["I_2:out0", "Add_1:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["batchnorm_single"],
"src_acu_in_tensor_map": [["I:out0", "batchnorm_single:in0"], ["I_1:out0", "batchnorm_single:in1"],
    ["I_2:out0", "batchnorm_single:in2"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnorm_single:out0"]],
"acu_inter_flow": [],
"param_map": {"batchnorm_single": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],}},
"blob_map": {"batchnorm_single": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                  'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_batchnorm_single)


r_batchnorm_single_1 = {
"ruler_name": "batchnorm_single_1",
"src_ops_alias": ["AddV2", "Mul", "Sub", "Mul_1", "C", "Mul_2", "Rsqrt", "C_1", "AddV2_1", "C_2"],
"src_inter_flow": [["Mul:out0", "AddV2:in0"], ["Sub:out0", "AddV2:in1"], ["Mul_1:out0", "Mul:in1"],
    ["C:out0", "Sub:in0"],
    ["Mul_2:out0", "Sub:in1"], ["Rsqrt:out0", "Mul_1:in0"], ["C_1:out0", "Mul_1:in1"],
    ["Mul_1:out0", "Mul_2:in1"], ["AddV2_1:out0", "Rsqrt:in0"], ["C_2:out0", "AddV2_1:in1"]],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I_1:out0", "Mul_2:in0"], ["I_2:out0", "AddV2_1:in0"]],
"src_out_tensor": ["AddV2:out0"],
"acu_lys_alias": ["batchnorm_single"],
"src_acu_in_tensor_map": [["I:out0", "batchnorm_single:in0"], ["I_1:out0", "batchnorm_single:in1"],
    ["I_2:out0", "batchnorm_single:in2"]],
"src_acu_out_tensor_map": [["AddV2:out0", "batchnorm_single:out0"]],
"acu_inter_flow": [],
"param_map": {"batchnorm_single": {'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],}},
"blob_map": {"batchnorm_single": {'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                  'scale': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_batchnorm_single_1)


@rule_pyfunc_def
def r_get_group_norm_group(self, node, tensor, name):
    _shape = list(self.shape_pick(tensor[name]))
    return _shape[-2]

@rule_pyfunc_def
def r_group_norm_pre_condition(self, node, tensor, name):
    _rank = len(self.shape_pick(tensor[name]))
    # Group input/output shape is [N,H,W,G,C/G]
    if _rank == 5:
        return True
    return False

r_group_norm = {
"ruler_name": "r_group_norm",
"src_ops_alias": ["Reshape", "AddV2", "C", "Mul", "Sub", "Reshape_1", "Mul_1", "C_1", "Mul_2", "C_2", "Rsqrt", "C_3",
    "Mean", "AddV2_1", "C_4", "Mean_1", "C_5", "SquaredDifference", "C_6", "StopGradient"],
"src_inter_flow": [["AddV2:out0", "Reshape:in0"], ["C:out0", "Reshape:in1"], ["Mul:out0", "AddV2:in0"],
    ["Sub:out0", "AddV2:in1"], ["Reshape_1:out0", "Mul:in0"], ["Mul_1:out0", "Mul:in1"],
    ["C_1:out0", "Sub:in0"], ["Mul_2:out0", "Sub:in1"], ["C_2:out0", "Reshape_1:in1"],
    ["Rsqrt:out0", "Mul_1:in0"], ["C_3:out0", "Mul_1:in1"], ["Mul_1:out0", "Mul_2:in1"],
    ["Mean:out0", "Mul_2:in0"], ["AddV2_1:out0", "Rsqrt:in0"], ["Reshape_1:out0", "Mean:in0"],
    ["C_4:out0", "Mean:in1"], ["Mean_1:out0", "AddV2_1:in0"], ["C_5:out0", "AddV2_1:in1"],
    ["SquaredDifference:out0", "Mean_1:in0"], ["C_6:out0", "Mean_1:in1"],
    ["Reshape_1:out0", "SquaredDifference:in0"], ["StopGradient:out0", "SquaredDifference:in1"],
    ["Mean:out0", "StopGradient:in0"]],
"src_in_anchor": [["I:out0", "Reshape_1:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["groupnormalize"],
"src_acu_in_tensor_map": [["I:out0", "groupnormalize:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "groupnormalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "groupnormalize": {
        'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_5:out0'])"],
        'num_groups': ['ORIGIN', 'PYFUNC', r_get_group_norm_group(name='Reshape_1:out0')],
    }
},
"blob_map": {
    "groupnormalize": {
        'scale': ['CODE', "self.tensor_to_numpy(tensor['C_3:out0'])"],
        'bias': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
    }
},
"priority_tip": 0,
"pre_condition": r_group_norm_pre_condition(name='Reshape_1:out0')}
ruler_list.append(r_group_norm)

r_gather = {
"ruler_name": "gather",
"src_ops_alias": ["Gather"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Gather:in0"], ["I_1:out0", "Gather:in1"]],
"src_out_tensor": ["Gather:out0"],
"acu_lys_alias": ["gather"],
"src_acu_in_tensor_map": [["I:out0", "gather:in0"], ["I_1:out0", "gather:in1"]],
"src_acu_out_tensor_map": [["Gather:out0", "gather:out0"]],
"acu_inter_flow": [],
"param_map": {"gather": {}},
"blob_map": {"gather": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather)

r_gather_axis = {
"ruler_name": "gather_axis",
"src_ops_alias": ["GatherV2", "C"],
"src_inter_flow": [["C:out0", "GatherV2:in2"]],
"src_in_anchor": [["I:out0", "GatherV2:in0"], ["I_1:out0", "GatherV2:in1"]],
"src_out_tensor": ["GatherV2:out0"],
"acu_lys_alias": ["gather"],
"src_acu_in_tensor_map": [["I:out0", "gather:in0"], ["I_1:out0", "gather:in1"]],
"src_acu_out_tensor_map": [["GatherV2:out0", "gather:out0"]],
"acu_inter_flow": [],
"param_map": {"gather": {'axis': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {"gather": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather_axis)

r_gather_nd = {
"ruler_name": "gather_nd",
"src_ops_alias": ["GatherNd"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GatherNd:in0"], ["I_1:out0", "GatherNd:in1"]],
"src_out_tensor": ["GatherNd:out0"],
"acu_lys_alias": ["gathernd"],
"src_acu_in_tensor_map": [["I:out0", "gathernd:in0"], ["I_1:out0", "gathernd:in1"]],
"src_acu_out_tensor_map": [["GatherNd:out0", "gathernd:out0"]],
"param_map": {"gathernd": {}},
"blob_map": {"gathernd": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather_nd)

r_scaternd = {
"ruler_name": "scatter_nd",
"src_ops_alias": ["ScatterNd", "C"],
"src_inter_flow": [["C:out0", "ScatterNd:in2"]],
"src_in_anchor": [["I:out0", "ScatterNd:in0"], ["I_1:out0", "ScatterNd:in1"]],
"src_out_tensor": ["ScatterNd:out0"],
"acu_lys_alias": ["scatternd"],
"src_acu_in_tensor_map": [["I:out0", "scatternd:in0"], ["I_1:out0", "scatternd:in1"]],
"src_acu_out_tensor_map": [["ScatterNd:out0", "scatternd:out0"]],
"param_map": {"scatternd": {'shape': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"],}},
"blob_map": {"scatternd": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_scaternd)

r_tile = {
"ruler_name": "tile",
"src_ops_alias": ["Tile", "C"],
"src_inter_flow": [["C:out0", "Tile:in1"]],
"src_in_anchor": [["I:out0", "Tile:in0"]],
"src_out_tensor": ["Tile:out0"],
"acu_lys_alias": ["tile"],
"src_acu_in_tensor_map": [["I:out0", "tile:in0"]],
"src_acu_out_tensor_map": [["Tile:out0", "tile:out0"]],
"param_map": {"tile": {'multiples': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"]}},
"blob_map": {"tile": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_tile)

r_repeat_2inputs_with_axis = {
"ruler_name": "repeat_2inputs_with_axis",
"src_ops_alias": ["GatherV2", "Reshape", "Squeeze", "C", "Tile", "C_1", "Where", "ExpandDims", "Pack", "Reshape_1",
    "C_2", "C_3", "Maximum", "C_4", "Less", "C_5", "C_6", "Max", "Range", "Cast", "BroadcastTo", "C_7",
    "C_8", "C_9", "ExpandDims_1", "Cast_1", "C_10", "C_11"],
"src_inter_flow": [["Reshape:out0", "GatherV2:in0"], ["Squeeze:out0", "GatherV2:in1"], ["C:out0", "GatherV2:in2"],
    ["Tile:out0", "Reshape:in0"], ["C_1:out0", "Reshape:in1"], ["Where:out0", "Squeeze:in0"],
    ["ExpandDims:out0", "Tile:in0"], ["Pack:out0", "Tile:in1"], ["Reshape_1:out0", "Where:in0"],
    ["C_2:out0", "ExpandDims:in1"], ["C_3:out0", "Pack:in0"], ["Maximum:out0", "Pack:in1"],
    ["C_4:out0", "Pack:in2"], ["Less:out0", "Reshape_1:in0"], ["C_5:out0", "Reshape_1:in1"],
    ["C_6:out0", "Maximum:in0"], ["Max:out0", "Maximum:in1"], ["Range:out0", "Less:in0"],
    ["Cast:out0", "Less:in1"], ["BroadcastTo:out0", "Max:in0"], ["C_7:out0", "Max:in1"],
    ["Maximum:out0", "Range:in1"], ["C_8:out0", "Range:in0"], ["C_9:out0", "Range:in2"],
    ["ExpandDims_1:out0", "Cast:in0"], ["Cast_1:out0", "BroadcastTo:in0"],
    ["C_10:out0", "BroadcastTo:in1"], ["BroadcastTo:out0", "ExpandDims_1:in0"],
    ["C_11:out0", "ExpandDims_1:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"], ["I_1:out0", "Cast_1:in0"]],
"src_out_tensor": ["GatherV2:out0"],
"acu_lys_alias": ["repeat"],
"src_acu_in_tensor_map": [["I:out0", "repeat:in0"], ["I_1:out0", "repeat:in1"]],
"src_acu_out_tensor_map": [["GatherV2:out0", "repeat:out0"]],
"acu_inter_flow": [],
"param_map": {"repeat": {'axis': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {"repeat": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_repeat_2inputs_with_axis)

r_sequencemask_with_2input = {
"ruler_name": "sequencemask_with_2input",
"src_ops_alias": ["Less", "Range", "Cast", "C", "C_1", "ExpandDims", "C_2"],
"src_inter_flow": [["Range:out0", "Less:in0"], ["Cast:out0", "Less:in1"], ["C:out0", "Range:in0"],
    ["C_1:out0", "Range:in2"], ["ExpandDims:out0", "Cast:in0"], ["C_2:out0", "ExpandDims:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"], ["I_1:out0", "Range:in1"]],
"src_out_tensor": ["Less:out0"],
"acu_lys_alias": ["sequence_mask"],
"src_acu_in_tensor_map": [["I:out0", "sequence_mask:in0"], ["I_1:out0", "sequence_mask:in1"]],
"src_acu_out_tensor_map": [["Less:out0", "sequence_mask:out0"]],
"acu_inter_flow": [],
"param_map": {"sequence_mask": {}},
"blob_map": {"sequence_mask": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_sequencemask_with_2input)

r_maxpool={
"ruler_name": "max_pool",
"src_ops_alias": ["MaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MaxPool:in0"]],
"src_out_tensor": ["MaxPool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["MaxPool:out0", "pooling:out0"]],
"param_map": {"pooling": {'padding': ['STRING', 'CODE', "self.attr_pick(node['MaxPool'], 'padding')"],
                            'round_type': ['STRING', 'VALUE', "floor"],
                            'type': ['STRING', 'VALUE', "MAX"],
                            'ksize_h':['INT', 'CODE', "self.attr_pick(node['MaxPool'], 'ksize')[1]"],
                            'ksize_w':['INT', 'CODE', "self.attr_pick(node['MaxPool'], 'ksize')[2]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['MaxPool'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['MaxPool'], 'strides')[2]"],
                            }},
"blob_map": {"pooling": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_maxpool)

r_avgpool = {
"ruler_name": "avgpool",
"src_ops_alias": ["AvgPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AvgPool:in0"]],
"src_out_tensor": ["AvgPool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["AvgPool:out0", "pooling:out0"]],
"param_map": {"pooling": {'padding': ['STRING', 'CODE', "self.attr_pick(node['AvgPool'], 'padding')"],
                            'round_type': ['STRING', 'VALUE', "floor"],
                            'type': ['STRING', 'VALUE', "AVG"],
                            'ksize_h':['INT', 'CODE', "self.attr_pick(node['AvgPool'], 'ksize')[1]"],
                            'ksize_w':['INT', 'CODE', "self.attr_pick(node['AvgPool'], 'ksize')[2]"],
                            'stride_h':['INT', 'CODE', "self.attr_pick(node['AvgPool'], 'strides')[1]"],
                            'stride_w':['INT', 'CODE', "self.attr_pick(node['AvgPool'], 'strides')[2]"],}},
"blob_map": {"pooling": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_avgpool)

# r_pad_avgpool = dict()
# r_pad_avgpool['src_ops_alias'] = ['Pad_1', 'AvgPool_2']
# r_pad_avgpool['acu_lys_alias'] = ['pooling_1']
# r_pad_avgpool['pre_condition'] = "self.attr_pick(tensor['AvgPool_2'], 'padding') == 'VALID'"
# r_pad_avgpool['param_map'] = {'pooling_1':
#                            {'padding': ['STRING', 'CODE', "self.attr_pick(tensor['AvgPool_2'], 'padding')"],
#                             'round_type': ['STRING', 'VALUE', "floor"],
#                             'type': ['STRING', 'VALUE', "AVG"],
#                             'ksize_h':['INT', 'CODE', "self.attr_pick(tensor['AvgPool_2'], 'ksize')[1]"],
#                             'ksize_w':['INT', 'CODE', "self.attr_pick(tensor['AvgPool_2'], 'ksize')[2]"],
#                             'stride_h':['INT', 'CODE', "self.attr_pick(tensor['AvgPool_2'], 'strides')[1]"],
#                             'stride_w':['INT', 'CODE', "self.attr_pick(tensor['AvgPool_2'], 'strides')[2]"],
#                             'pad_h': ['INT', 'CODE', "self.tensor_to_numpy(tensor['Pad_1'].input[1])[1][0]"],
#                             'pad_w': ['INT', 'CODE', "self.tensor_to_numpy(tensor['Pad_1'].input[1])[2][0]"],
#                             }
#                     }
# ruler_list.append(r_pad_avgpool)

r_reducemean2avgpool = {
"ruler_name": "reduce_mean_keepshape",
"src_ops_alias": ["Mean", "C"],
"src_inter_flow": [["C:out0", "Mean:in1"]],
"src_in_anchor": [["I:out0", "Mean:in0"]],
"src_out_tensor": ["Mean:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["Mean:out0", "pooling:out0"]],
"param_map": {"pooling":  {'padding': ['STRING', 'VALUE', "VALID"],
                                                'round_type': ['STRING', 'VALUE', "floor"],
                                                'type': ['STRING', 'VALUE', "AVG"],
                                                'stride_h':['INT', 'VALUE', 1],
                                                'stride_w':['INT', 'VALUE', 1],
                                                'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['I:out0'])[1]"],
                                                'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['I:out0'])[2]"]
                                                },},
"blob_map": {"pooling": {}},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": "self.attr_pick(node['Mean'], 'keep_dims') == True "\
                 "and self.tensor_to_numpy(tensor['C:out0']).tolist() == [1, 2]"}  # make sure tensor is [1, 2]
ruler_list.append(r_reducemean2avgpool)

r_reducemean2avgpoolreshape = {
"ruler_name": "reduce_mean_nokeepshape",
"src_ops_alias": ["Mean", "C"],
"src_inter_flow": [["C:out0", "Mean:in1"]],
"src_in_anchor": [["I:out0", "Mean:in0"]],
"src_out_tensor": ["Mean:out0"],
"acu_lys_alias": ["pooling", "reshape"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["Mean:out0", "reshape:out0"]],
"param_map": {"reshape": {'shape': ['INTS', 'VALUE', [0, -1]],},
              "pooling":  {'padding': ['STRING', 'VALUE', "VALID"],
                                                'round_type': ['STRING', 'VALUE', "floor"],
                                                'type': ['STRING', 'VALUE', "AVG"],
                                                'stride_h':['INT', 'VALUE', 1],
                                                'stride_w':['INT', 'VALUE', 1],
                                                'ksize_h': ['INT', 'CODE', "self.shape_pick(tensor['I:out0'])[1]"],
                                                'ksize_w': ['INT', 'CODE', "self.shape_pick(tensor['I:out0'])[2]"]
                                                },},
"blob_map": {"reshape": {}, "pooling": {}},
"acu_inter_flow": [["pooling:out0", "reshape:in0"]],
"priority_tip": 1,
"pre_condition": "self.attr_pick(node['Mean'], 'keep_dims') == False "\
                 "and self.tensor_to_numpy(tensor['C:out0']).tolist() == [1, 2]"}  # make sure tensor is [1, 2]
ruler_list.append(r_reducemean2avgpoolreshape)

@rule_pyfunc_def
def r_reduce_x_get_axis(self, node, tensor, axis_tensor_name):
    axis = self.tensor_to_numpy(tensor[axis_tensor_name]).tolist()
    if isinstance(axis,list):
        return axis
    else:
        return [axis]
r_reducemean = {
"ruler_name": "reducemean",
"src_ops_alias": ["Mean", "C"],
"src_inter_flow": [["C:out0", "Mean:in1"]],
"src_in_anchor": [["I:out0", "Mean:in0"]],
"src_out_tensor": ["Mean:out0"],
"acu_lys_alias": ["reducemean"],
"src_acu_in_tensor_map": [["I:out0", "reducemean:in0"]],
"src_acu_out_tensor_map": [["Mean:out0", "reducemean:out0"]],
"param_map": {
              "reducemean":  {'axis_list': ['ORIGIN', 'PYFUNC', r_reduce_x_get_axis(axis_tensor_name='C:out0')],
                             'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['Mean'], 'keep_dims')"]}
            },
"blob_map": {"reducemean": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_reducemean)

r_reducemax = {
"ruler_name": "r_reducemax",
"src_ops_alias": ["Max", "C"],
"src_inter_flow": [["C:out0", "Max:in1"]],
"src_in_anchor": [["I:out0", "Max:in0"]],
"src_out_tensor": ["Max:out0"],
"acu_lys_alias": ["reducemax"],
"src_acu_in_tensor_map": [["I:out0", "reducemax:in0"]],
"src_acu_out_tensor_map": [["Max:out0", "reducemax:out0"]],
"param_map": {
              "reducemax":  {'axis_list': ['ORIGIN', 'PYFUNC', r_reduce_x_get_axis(axis_tensor_name='C:out0')],
                             'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['Max'], 'keep_dims')"]}
            },
"blob_map": {"reducemax": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_reducemax)

r_batchtospace = {
"ruler_name": "batch_to_space",
"src_ops_alias": ["BatchToSpaceND", "C", "C_1"],
"src_inter_flow": [["C:out0", "BatchToSpaceND:in1"], ["C_1:out0", "BatchToSpaceND:in2"]],
"src_in_anchor": [["I:out0", "BatchToSpaceND:in0"]],
"src_out_tensor": ["BatchToSpaceND:out0"],
"acu_lys_alias": ["batch2space"],
"src_acu_in_tensor_map": [["I:out0", "batch2space:in0"]],
"src_acu_out_tensor_map": [["BatchToSpaceND:out0", "batch2space:out0"]],
"acu_inter_flow": [],
"param_map": {"batch2space": {'block_shape': ['INTS','CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                              'block_crops': ['ORIGIN','CODE', "self.tensor_to_numpy(tensor['C_1:out0']).tolist()"]}},
"blob_map": {"batch2space": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_batchtospace)

r_spacetobatch = {
"ruler_name": "space_to_batch",
"src_ops_alias": ["SpaceToBatchND", "C", "C_1"],
"src_inter_flow": [["C:out0", "SpaceToBatchND:in1"], ["C_1:out0", "SpaceToBatchND:in2"]],
"src_in_anchor": [["I:out0", "SpaceToBatchND:in0"]],
"src_out_tensor": ["SpaceToBatchND:out0"],
"acu_lys_alias": ["space2batch"],
"src_acu_in_tensor_map": [["I:out0", "space2batch:in0"]],
"src_acu_out_tensor_map": [["SpaceToBatchND:out0", "space2batch:out0"]],
"acu_inter_flow": [],
"param_map": {"space2batch": {
                             'block_shape': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                             'block_paddings': ['ORIGIN','CODE', "self.tensor_to_numpy(tensor['C_1:out0']).tolist()"]
                              }},
"blob_map": {"space2batch": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_spacetobatch)

r_poolwithargmax = {
"ruler_name": "poolwithargmax",
"src_ops_alias": ["MaxPoolWithArgmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MaxPoolWithArgmax:in0"]],
"src_out_tensor": ["MaxPoolWithArgmax:out0"],
"acu_lys_alias": ["poolwithargmax"],
"src_acu_in_tensor_map": [["I:out0", "poolwithargmax:in0"]],
"src_acu_out_tensor_map": [["MaxPoolWithArgmax:out0", "poolwithargmax:out0"]],
"param_map": {"poolwithargmax": {}},
"blob_map": {"poolwithargmax": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_poolwithargmax)

r_rsp_v1 = {
"ruler_name": "reshape",
"src_ops_alias": ["Reshape", "C"],
"src_inter_flow": [["C:out0", "Reshape:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C:out0'])"],}},
"blob_map": {"reshape": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_rsp_v1)

@rule_pyfunc_def
def r_get_expand_dim_out_shape(self, node, tensor, axis_tensor_name):
    input_shape = list(self.shape_pick(tensor['I:out0']))
    input_shape_dims = len(input_shape)
    axis = self.tensor_to_numpy(tensor['C:out0'])
    # convert netagive axis to positive to calculate output shape easier
    if axis[0] < 0:
        axis[0] = (axis[0] + (input_shape_dims + 1)) % (input_shape_dims + 1)
    output_shape = input_shape[0:axis[0]] + [1] + input_shape[axis[0]:]
    return output_shape

r_expand_dim = {
"ruler_name": "expand_dims",
"src_ops_alias": ["ExpandDims", "C"],
"src_inter_flow": [["C:out0", "ExpandDims:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"]],
"src_out_tensor": ["ExpandDims:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["ExpandDims:out0", "reshape:out0"]],
"param_map":
    {"reshape":
         {'shape':
              ['ORIGIN', 'PYFUNC', r_get_expand_dim_out_shape(axis_tensor_name='C:out0')],
          }
     },
"blob_map": {"reshape": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_expand_dim)

r_squeeze = {
"ruler_name": "squeeze",
"src_ops_alias": ["Squeeze"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "reshape:out0"]],
"param_map":
    {"reshape":
         {'shape':
              ['INTS',
               'CODE',
               "self.squeeze_shapes("\
               "self.attr_pick(node['Squeeze'], "\
               "'squeeze_dims', None), "\
               "self.shape_pick(tensor['I:out0']))"
               ]
          }
     },
"blob_map": {"reshape": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_squeeze)

r_transpose = {
"ruler_name": "transpose",
"src_ops_alias": ["Transpose", "C"],
"src_inter_flow": [["C:out0", "Transpose:in1"]],
"src_in_anchor": [["I:out0", "Transpose:in0"]],
"src_out_tensor": ["Transpose:out0"],
"acu_lys_alias": ["permute"],
"src_acu_in_tensor_map": [["I:out0", "permute:in0"]],
"src_acu_out_tensor_map": [["Transpose:out0", "permute:out0"]],
"param_map": {
    "permute": {'perm': ['STRING', 'CODE', "' '.join([str(perm) for perm in self.tensor_to_numpy(tensor['C:out0'])])"],
                }
},
"blob_map": {"permute": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_transpose)

r_transpose_to_noop = {
"ruler_name": "transpose_to_noop",
"src_ops_alias": ["Transpose", "C"],
"src_inter_flow": [["C:out0", "Transpose:in1"]],
"src_in_anchor": [["I:out0", "Transpose:in0"]],
"src_out_tensor": ["Transpose:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Transpose:out0", "noop:out0"]],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition":
    "self.tensor_to_numpy(tensor['C:out0']).tolist() "\
    "== [ i for i in range(len(self.tensor_to_numpy(tensor['C:out0'])))]"}
ruler_list.append(r_transpose_to_noop)

r_depth2space = {
"ruler_name": "depth2space",
"src_ops_alias": ["DepthToSpace"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "DepthToSpace:in0"]],
"src_out_tensor": ["DepthToSpace:out0"],
"acu_lys_alias": ["depth2space"],
"src_acu_in_tensor_map": [["I:out0", "depth2space:in0"]],
"src_acu_out_tensor_map": [["DepthToSpace:out0", "depth2space:out0"]],
"param_map": {"depth2space": {'block_size': ['INT', 'CODE', "self.attr_pick(node['DepthToSpace'], 'block_size')"]}},
"blob_map": {"depth2space": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_depth2space)

r_space2depth = {
"ruler_name": "space2depth",
"src_ops_alias": ["SpaceToDepth"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "SpaceToDepth:in0"]],
"src_out_tensor": ["SpaceToDepth:out0"],
"acu_lys_alias": ["space2depth"],
"src_acu_in_tensor_map": [["I:out0", "space2depth:in0"]],
"src_acu_out_tensor_map": [["SpaceToDepth:out0", "space2depth:out0"]],
"param_map": {"space2depth": {'block_size': ['INT', 'CODE', "self.attr_pick(node['SpaceToDepth'], 'block_size')"]}},
"blob_map": {"space2depth": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_space2depth)

r_reverse = {
"ruler_name": "reverse",
"src_ops_alias": ["ReverseV2", "C"],
"src_inter_flow": [["C:out0", "ReverseV2:in1"]],
"src_in_anchor": [["I:out0", "ReverseV2:in0"]],
"src_out_tensor": ["ReverseV2:out0"],
"acu_lys_alias": ["reverse"],
"src_acu_in_tensor_map": [["I:out0", "reverse:in0"]],
"src_acu_out_tensor_map": [["ReverseV2:out0", "reverse:out0"]],
"param_map": {"reverse": {'axis':['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"blob_map": {"reverse": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_reverse)

r_pad = {
"ruler_name": "pad",
"src_ops_alias": ["Pad", "C"],
"src_inter_flow": [["C:out0", "Pad:in1"]],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"param_map": {"pad": {'padding_value': ['ORIGIN', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"],
                      'padding_mode': ['ORIGIN', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'CONSTANT')"]}},
"blob_map": {"pad": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_pad)

r_mirrorpad = {
"ruler_name": "mirrorpad",
"src_ops_alias": ["MirrorPad", "C"],
"src_inter_flow": [["C:out0", "MirrorPad:in1"]],
"src_in_anchor": [["I:out0", "MirrorPad:in0"]],
"src_out_tensor": ["MirrorPad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["MirrorPad:out0", "pad:out0"]],
"param_map": {"pad": {'padding_value': ['ORIGIN', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"],
                      'padding_mode': ['ORIGIN', 'CODE', "self.attr_pick(node['MirrorPad'], 'mode', 'CONSTANT')"]}},
"blob_map": {"pad": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mirrorpad)

r_resizebilinear = {
"ruler_name": "resieze_biliner",
"src_ops_alias": ["ResizeBilinear", "C"],
"src_inter_flow": [["C:out0", "ResizeBilinear:in1"]],
"src_in_anchor": [["I:out0", "ResizeBilinear:in0"]],
"src_out_tensor": ["ResizeBilinear:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["ResizeBilinear:out0", "image_resize:out0"]],
"param_map": {"image_resize": {'new_size': ['ORIGIN', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"],
                            'align_corners':
                                ['BOOL', 'CODE', "self.attr_pick(node['ResizeBilinear'], 'align_corners')"],
                            'type': ['STRING', 'VALUE', 'bilinear'],}},
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_resizebilinear)

r_resizenearest ={
"ruler_name": "resizenearest",
"src_ops_alias": ["ResizeNearestNeighbor", "C"],
"src_inter_flow": [["C:out0", "ResizeNearestNeighbor:in1"]],
"src_in_anchor": [["I:out0", "ResizeNearestNeighbor:in0"]],
"src_out_tensor": ["ResizeNearestNeighbor:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["ResizeNearestNeighbor:out0", "image_resize:out0"]],
"param_map": {"image_resize": {'new_size': ['ORIGIN', 'CODE', "self.tensor_to_numpy(tensor['C:out0']).tolist()"],
                            'align_corners':
                                [
                                    'BOOL',
                                    'CODE',
                                    "self.attr_pick(node['ResizeNearestNeighbor'], 'align_corners', False)"
                                ],
                            'type': ['STRING', 'VALUE', 'nearest'],}},
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_resizenearest)

r_reshape_mul_reshape_to_upsampling = {
"ruler_name": "upsampling",
"src_ops_alias": ["Reshape", "Mul", "C", "Reshape_1", "C_1", "C_2"],
"src_inter_flow": [["Reshape_1:out0", "Mul:in0"], ["C_2:out0", "Reshape_1:in1"], ["C:out0", "Reshape:in1"],
                   ["Mul:out0", "Reshape:in0"], ["C_1:out0", "Mul:in1"]],
"src_in_anchor": [["I:out0", "Reshape_1:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["upsampling"],
"src_acu_in_tensor_map": [["I:out0", "upsampling:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "upsampling:out0"]],
"param_map": {"upsampling": {'factor': ['ORIGIN', 'CODE', "self.shape_pick(tensor['C_1:out0'])[2]"],}},
"blob_map": {"upsampling": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['C_1:out0']) == np.ones([1, 1, 2, 1, 2, 1])).all()"}
ruler_list.append(r_reshape_mul_reshape_to_upsampling)

def r_concat_template(concat_count, const_count=1):
    r_concat_dict = {
"ruler_name": "concat_{}".format(concat_count),
"src_ops_alias": ["Concat", "C_0"],
"src_inter_flow": [["C_{}:out0".format(n), "Concat:in{}".format(n)] for n in range(const_count)],
"src_in_anchor": [["I_{}:out0".format(order - const_count), "Concat:in{}".format(order)]
                  for order in range(const_count, const_count + concat_count)],
"src_out_tensor": ["Concat:out0"],
"acu_lys_alias": ["concat"],
"src_acu_in_tensor_map": [["I_{}:out0".format(order), "concat:in{}".format(order)]
                          for order in range(concat_count)],
"src_acu_out_tensor_map": [["Concat:out0", "concat:out0"]],
"param_map": {"concat": {'dim': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_0:out0'])"]}},
"blob_map": {"concat": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
    return r_concat_dict

def r_concat_specail_template(concat_count, concat_input_list, const_count=1):
    r_concat_specail_dict = {
"ruler_name": "concat_specail_{}".format(concat_count),
"src_ops_alias": ["Concat", "C_0"],
"src_inter_flow": [["C_{}:out0".format(n), "Concat:in{}".format(n)] for n in range(const_count)],
"src_in_anchor": [["I_{}:out0".format(concat_input_list[order - const_count]), "Concat:in{}".format(order)]
                  for order in range(const_count, const_count + concat_count)],
"src_out_tensor": ["Concat:out0"],
"acu_lys_alias": ["concat"],
"src_acu_in_tensor_map": [["I_{}:out0".format(concat_input_list[order]), "concat:in{}".format(order)]
                          for order in range(concat_count)],
"src_acu_out_tensor_map": [["Concat:out0", "concat:out0"]],
"param_map": {"concat": {'dim': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_0:out0'])"]}},
"blob_map": {"concat": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
    return r_concat_specail_dict

def r_concatv2_template(concat_count):
    r_concat_dict = {
"ruler_name": "concatv2_{}".format(concat_count),
"src_ops_alias": ["ConcatV2", "C"],
"src_inter_flow": [["C:out0", "ConcatV2:in{}".format(concat_count)]],
"src_in_anchor": [["I_{}:out0".format(order), "ConcatV2:in{}".format(order)] for order in range(concat_count)],
"src_out_tensor": ["ConcatV2:out0"],
"acu_lys_alias": ["concat"],
"src_acu_in_tensor_map": [["I_{}:out0".format(order), "concat:in{}".format(order)] for order in range(concat_count)],
"src_acu_out_tensor_map": [["ConcatV2:out0", "concat:out0"]],
"param_map": {"concat": {'dim': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {"concat": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
    return r_concat_dict

def r_concatv2_specail_template(concatv2_count, concatv2_input_list):
    r_concatv2_specail_dict = {
"ruler_name": "concatv2_specail_{}".format(concatv2_count),
"src_ops_alias": ["ConcatV2", "C"],
"src_inter_flow": [["C:out0", "ConcatV2:in{}".format(concatv2_count)]],
"src_in_anchor": [["I_{}:out0".format(concatv2_input_list[order]), "ConcatV2:in{}".format(order)]
                  for order in range(concatv2_count)],
"src_out_tensor": ["ConcatV2:out0"],
"acu_lys_alias": ["concat"],
"src_acu_in_tensor_map": [["I_{}:out0".format(concatv2_input_list[order]), "concat:in{}".format(order)]
                          for order in range(concatv2_count)],
"src_acu_out_tensor_map": [["ConcatV2:out0", "concat:out0"]],
"param_map": {"concat": {'dim': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {"concat": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
    return r_concatv2_specail_dict

#for i in range(2, 12):
#    ruler_list.append(r_concatv2_template(i))
#ruler_list.append(r_concatv2_template(80))

#the concat_strip_in1 ruler is for unreasonable model
#the shape of in1 is 0.
r_concat_strip_in1 = {
"ruler_name": "concat_strip_in1",
"src_ops_alias": ["ConcatV2", "C_1", "C_2"],
"src_inter_flow": [["C_1:out0", "ConcatV2:in1"], ["C_2:out0", "ConcatV2:in2"]],
"src_in_anchor": [["I_0:out0", "ConcatV2:in0"]],
"src_out_tensor": ["ConcatV2:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ConcatV2:out0", "noop:out0"]],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": "0 in self.shape_pick(tensor['C_1:out0'])"}
ruler_list.append(r_concat_strip_in1)

def r_stack_template(stack_count):
    r_stack_dict = {
        "ruler_name": "stack_{}".format(stack_count),
        "src_ops_alias": ["Pack"],
        "src_inter_flow": [],
        "src_in_anchor":
            [["I_{}:out0".format(order), "Pack:in{}".format(order)] for order in range(stack_count)],
        "src_out_tensor": ["Pack:out0"],
        "acu_lys_alias": ["stack"],
        "src_acu_in_tensor_map":
            [["I_{}:out0".format(order), "stack:in{}".format(order)] for order in range(stack_count)],
        "src_acu_out_tensor_map": [["Pack:out0", "stack:out0"]],
        "param_map": {"stack":
                          {'axis': ['INT', 'CODE', "self.attr_pick(node['Pack'], 'axis', 0)"]}
                      },
        "blob_map": {"stack": {}},
        "acu_inter_flow": [],
        "priority_tip": 0,
        "pre_condition": None}
    return r_stack_dict

#for i in range(1, 12):
#    ruler_list.append(r_stack_template(i))

def r_stack_specail_template(stack_count, stack_input_list):
    r_stack_specail_dict = {
        "ruler_name": "stack_specail_{}".format(stack_count),
        "src_ops_alias": ["Pack"],
        "src_inter_flow": [],
        "src_in_anchor":
            [["I_{}:out0".format(stack_input_list[order]), "Pack:in{}".format(order)] for order in range(stack_count)],
        "src_out_tensor": ["Pack:out0"],
        "acu_lys_alias": ["stack"],
        "src_acu_in_tensor_map":
        [["I_{}:out0".format(stack_input_list[order]), "stack:in{}".format(order)] for order in range(stack_count)],
        "src_acu_out_tensor_map": [["Pack:out0", "stack:out0"]],
        "param_map": {"stack":
                          {'axis': ['INT', 'CODE', "self.attr_pick(node['Pack'], 'axis', 0)"]}
                      },
        "blob_map": {"stack": {}},
        "acu_inter_flow": [],
        "priority_tip": 0,
        "pre_condition": None}
    return r_stack_specail_dict


def r_split_template(split_count):
    r_split_dict = {
        "ruler_name": "split{}".format(split_count),
        "src_ops_alias": ["Split", "C"],
        "src_inter_flow": [["C:out0", "Split:in0"]],
        "src_in_anchor": [["I:out0", "Split:in1"]],
        "src_out_tensor": ["Split:out{}".format(order) for order in range(split_count)],
        "acu_lys_alias": ["split"],
        "src_acu_in_tensor_map": [["I:out0", "split:in0"]],
        "src_acu_out_tensor_map":
            [["Split:out{}".format(order), "split:out{}".format(order)] for order in range(split_count)],
        "param_map": {"split": {'dim': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                                'slices':
                                    ['INTS',
                                     'CODE',
                                     "self.split_slice("\
                                     "self.tensor_to_numpy(tensor['C:out0']),"\
                                     " self.shape_pick(tensor['I:out0']), "\
                                     "self.attr_pick(node['Split'], 'num_split', 1))"
                                     ],
                                }},
        "blob_map": {"split": {}},
        "acu_inter_flow": [],
        "priority_tip": 0,
        "pre_condition": None}
    return r_split_dict

#ruler_list.append(r_split_template(2))
#ruler_list.append(r_split_template(4))

def r_splitv_template(split_count):
    r_splitv_dict = {
        "ruler_name": "splitv{}",
        "src_ops_alias": ["SplitV", "C", "C_1"],
        "src_inter_flow": [["C:out0", "SplitV:in1"], ["C_1:out0", "SplitV:in2"]],
        "src_in_anchor": [["I:out0", "SplitV:in0"]],
        "src_out_tensor": ["SplitV:out{}".format(order) for order in range(split_count)],
        "acu_lys_alias": ["split"],
        "src_acu_in_tensor_map": [["I:out0", "split:in0"]],
        "src_acu_out_tensor_map":
            [["SplitV:out{}".format(order), "split:out{}".format(order)] for order in range(split_count)],
        "acu_inter_flow": [],
        "param_map": {"split": {'dim': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                                'slices':
                                    ['INTS',
                                     'CODE',
                                     "self.splitv_slice("\
                                     "self.tensor_to_numpy(tensor['C:out0']),"\
                                     "self.attr_pick(node['SplitV'], 'num_split', 1))"
                                     ],
                                }},
        "blob_map": {"split": {}},
        "priority_tip": 0,
        "pre_condition": None}
    return r_splitv_dict


def r_unpack_template(split_count):
    r_split_dict = {
        "ruler_name": "unpack_{}".format(split_count),
        "src_ops_alias": ["Unpack"],
        "src_inter_flow": [],
        "src_in_anchor": [["I:out0", "Unpack:in0"]],
        "src_out_tensor": ["Unpack:out{}".format(order) for order in range(split_count)],
        "acu_lys_alias": ["unstack"],
        "src_acu_in_tensor_map": [["I:out0", "unstack:in0"]],
        "src_acu_out_tensor_map":
            [["Unpack:out{}".format(order), "unstack:out{}".format(order)] for order in range(split_count)],
        "param_map": {"unstack": {'axis': ['INT', 'CODE', "self.attr_pick(node['Unpack'], 'axis', 1)"]}},
        "blob_map": {"unstack": {}},
        "acu_inter_flow": [],
        "priority_tip": 0,
        "pre_condition": None}
    return r_split_dict
#ruler_list.append(r_unpack_template(28))

def r_custom_template(op_name, inputs_cnt, outputs_cnt, out_port_list):
    r_custom_dict = {
        "ruler_name": "custom_{}_in{}_out{}".format(op_name, inputs_cnt, outputs_cnt),
        "src_ops_alias": ["{}".format(op_name)],
        "src_inter_flow": [],
        "src_in_anchor":
            [["I_{}:out0".format(order), "{}:in{}".format(op_name, order)] for order in range(inputs_cnt)],
        "src_out_tensor": ["{}:out{}".format(op_name, order) for order in out_port_list],
        "acu_lys_alias": ["customlayer"],
        "src_acu_in_tensor_map":
            [["I_{}:out0".format(order), "customlayer:in{}".format(order)] for order in range(inputs_cnt)],
        "src_acu_out_tensor_map":
            [["{}:out{}".format(op_name, src_order), "customlayer:out{}".format(acu_order)]
             for (src_order, acu_order) in zip(out_port_list, range(outputs_cnt))],
        "param_map": {"customlayer": {}},
        "blob_map": {"customlayer": {}},
        "acu_inter_flow": [],
        "priority_tip": 0,
        "pre_condition": None}
    return r_custom_dict

def r_while_custom_template(while_dict, while_block_idx):
    ruler_pre = while_dict['ruler_prepare']
    enter_cnt = len(ruler_pre['enter_ops_alias'])
    r_while_custom_dict = {
        "ruler_name": "custom_while{}".format(while_block_idx),
        "src_ops_alias": ["{}".format(op_alias) for op_alias in iter(ruler_pre['ops_alias'])],
        "src_inter_flow": [tensor_map for tensor_map in iter(ruler_pre['op_alias_maps'])],
        "src_in_anchor":
            [["I_{}:out0".format(idx),
            "{}:in0".format(enter_op)] for idx, enter_op in enumerate(ruler_pre['enter_ops_alias'])],
        "src_out_tensor": ["{}:out0".format(exit_op) for exit_op in iter(ruler_pre['exit_ops_alias'])],
        "acu_lys_alias": ["customlayer"],
        "src_acu_in_tensor_map":
            [["I_{}:out0".format(idx), "customlayer:in{}".format(idx)] for idx in range(enter_cnt)],
        "src_acu_out_tensor_map":
            [["{}:out0".format(exit_op), "customlayer:out{}".format(idx)]
             for idx, exit_op in enumerate(ruler_pre['exit_ops_alias'])],
        "param_map": {"customlayer": {}},
        "blob_map": {"customlayer": {}},
        "acu_inter_flow": [],
        "priority_tip": 0,
        "pre_condition": None}
    return r_while_custom_dict

r_split2noop = {
"ruler_name": "split",
"src_ops_alias": ["Split", "C"],
"src_inter_flow": [["C:out0", "Split:in0"]],
"src_in_anchor": [["I:out0", "Split:in1"]],
"src_out_tensor": ["Split:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Split:out0", "noop:out0"]],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": "self.attr_pick(node['Split'], 'num_split') == 1"}
ruler_list.append(r_split2noop)

r_stridedslice = {
"ruler_name": "stride_slice",
"src_ops_alias": ["StridedSlice", "C", "C_1", "C_2"],
"src_inter_flow": [["C:out0", "StridedSlice:in1"], ["C_2:out0", "StridedSlice:in3"], ["C_1:out0", "StridedSlice:in2"]],
"src_in_anchor": [["I:out0", "StridedSlice:in0"]],
"src_out_tensor": ["StridedSlice:out0"],
"acu_lys_alias": ["stridedslice"],
"src_acu_in_tensor_map": [["I:out0", "stridedslice:in0"]],
"src_acu_out_tensor_map": [["StridedSlice:out0", "stridedslice:out0"]],
"param_map": {"stridedslice":
                  {'slice_begin': ['INTS', 'CODE', "[int(p) for p in self.tensor_to_numpy(tensor['C:out0'])]"],
                    'slice_end': ['INTS', 'CODE', "[int(p) for p in self.tensor_to_numpy(tensor['C_1:out0'])]"],
                    'slice_strides': ['INTS', 'CODE', "[int(p) for p in self.tensor_to_numpy(tensor['C_2:out0'])]"],
                    'slice_begin_mask': ['INT', 'CODE',"self.attr_pick(node['StridedSlice'], 'begin_mask', 0)"],
                    'slice_end_mask': ['INT', 'CODE',"self.attr_pick(node['StridedSlice'], 'end_mask', 0)"],
                    'slice_ellipsis_mask': ['INT', 'CODE',"self.attr_pick(node['StridedSlice'], 'ellipsis_mask', 0)"],
                    'slice_new_axis_mask': ['INT', 'CODE',"self.attr_pick(node['StridedSlice'], 'new_axis_mask', 0)"],
                    'slice_shrink_axis_mask':
                       ['INT', 'CODE',"self.attr_pick(node['StridedSlice'], 'shrink_axis_mask', 0)"],
                   }
              },
"blob_map": {"stridedslice": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_stridedslice)

r_slice = {
"ruler_name": "slice",
"src_ops_alias": ["Slice", "C", "C_1"],
"src_inter_flow": [["C_1:out0", "Slice:in2"], ["C:out0", "Slice:in1"]],
"src_in_anchor": [["I:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"param_map": {"slice": {'begin': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                            'size': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"blob_map": {"slice": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_slice)

r_slice_two_inputs = {
"ruler_name": "r_slice_two_inputs",
"src_ops_alias": ["Slice", "C",],
"src_inter_flow": [["C:out0", "Slice:in2"]],
"src_in_anchor": [["I:out0", "Slice:in0"], ["I_1:out0", "Slice:in1"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I:out0", "slice:in0"], ["I_1:out0", "slice:in1"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"param_map": {"slice": {'begin': ['ORIGIN', 'VALUE', []],
                            'size': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"blob_map": {"slice": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_slice_two_inputs)

r_softmax = {
"ruler_name": "softmax",
"src_ops_alias": ["Softmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Softmax:in0"]],
"src_out_tensor": ["Softmax:out0"],
"acu_lys_alias": ["softmax"],
"src_acu_in_tensor_map": [["I:out0", "softmax:in0"]],
"src_acu_out_tensor_map": [["Softmax:out0", "softmax:out0"]],
"param_map": {"softmax": {}},
"blob_map": {"softmax": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_softmax)

r_softmax_formal = {
"ruler_name": "softmax_calc",
"src_ops_alias": ["RealDiv", "Exp", "Sum", "Sub", "C", "Max", "C_1"],
"src_inter_flow": [["Sum:out0", "RealDiv:in1"], ["Exp:out0", "RealDiv:in0"], ["Max:out0", "Sub:in1"],
                   ["Exp:out0", "Sum:in0"], ["C:out0", "Sum:in1"], ["Sub:out0", "Exp:in0"], ["C_1:out0", "Max:in1"]],
"src_in_anchor": [["I:out0", "Max:in0"], ["I:out0", "Sub:in0"]],
"src_out_tensor": ["RealDiv:out0"],
"acu_lys_alias": ["softmax"],
"src_acu_in_tensor_map": [["I:out0", "softmax:in0"]],
"src_acu_out_tensor_map": [["RealDiv:out0", "softmax:out0"]],
"param_map": {"softmax": {'sf_axis':['INT', 'VALUE', -1],}},
"blob_map": {"softmax": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_softmax_formal)

r_logsoftmax = {
    "ruler_name": "log_softmax",
    "src_ops_alias": ["LogSoftmax"],
    "src_inter_flow": [],
    "src_in_anchor": [["I:out0", "LogSoftmax:in0"]],
    "src_out_tensor": ["LogSoftmax:out0"],
    "acu_lys_alias": ["log_softmax"],
    "src_acu_in_tensor_map": [["I:out0", "log_softmax:in0"]],
    "src_acu_out_tensor_map": [["LogSoftmax:out0", "log_softmax:out0"]],
    "param_map": {"log_softmax": {}},
    "blob_map": {"log_softmax": {}},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_logsoftmax)

r_mish = {
"ruler_name": "mish",
"src_ops_alias": ["Mul", "Tanh", "Softplus"],
"src_inter_flow": [["Tanh:out0", "Mul:in1"], ["Softplus:out0", "Tanh:in0"]],
"src_in_anchor": [["I:out0", "Softplus:in0"], ["I:out0", "Mul:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["mish"],
"src_acu_in_tensor_map": [["I:out0", "mish:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "mish:out0"]],
"acu_inter_flow": [],
"param_map": {"mish": {}},
"blob_map": {"mish": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mish)

r_mish_1 = {
"ruler_name": "mish_1",
"src_ops_alias": ["Mul", "Tanh", "Softplus"],
"src_inter_flow": [["Tanh:out0", "Mul:in0"], ["Softplus:out0", "Tanh:in0"]],
"src_in_anchor": [["I:out0", "Softplus:in0"], ["I:out0", "Mul:in1"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["mish"],
"src_acu_in_tensor_map": [["I:out0", "mish:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "mish:out0"]],
"acu_inter_flow": [],
"param_map": {"mish": {}},
"blob_map": {"mish": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mish_1)

r_rank = {
    "ruler_name": "rank",
    "src_ops_alias": ["Rank"],
    "src_inter_flow": [],
    "src_in_anchor": [["I:out0", "Rank:in0"]],
    "src_out_tensor": ["Rank:out0"],
    "acu_lys_alias": ["rank"],
    "src_acu_in_tensor_map": [["I:out0", "rank:in0"]],
    "src_acu_out_tensor_map": [["Rank:out0", "rank:out0"]],
    "param_map": {"rank": {}},
    "blob_map": {"rank": {}},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_rank)

r_add = {
"ruler_name": "add",
"src_ops_alias": ["Add"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Add:in0"], ["I_1:out0", "Add:in1"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map": [["I:out0", "add:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [["Add:out0", "add:out0"]],
"param_map": {"add": {}},
"blob_map": {"add": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_add)

r_addv2 = {
"ruler_name": "r_addv2",
"src_ops_alias": ["AddV2"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AddV2:in0"], ["I_1:out0", "AddV2:in1"]],
"src_out_tensor": ["AddV2:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map": [["I:out0", "add:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [["AddV2:out0", "add:out0"]],
"param_map": {"add": {}},
"blob_map": {"add": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_addv2)

# r_add_quant = {
# "ruler_name": "add_quant",
# "src_ops_alias": ["Add", "FakeQuantWithMinMaxVars", "C", "C_1"],
# "src_inter_flow": [["Add:out0","FakeQuantWithMinMaxVars:in0"],["C:out0","FakeQuantWithMinMaxVars:in1"],
#                    ["C_1:out0","FakeQuantWithMinMaxVars:in2"]],
# "src_in_anchor": [["I:out0", "Add:in0"], ["I_1:out0", "Add:in1"]],
# "src_out_tensor": ["FakeQuantWithMinMaxVars:out0"],
# "acu_lys_alias": ["add"],
# "src_acu_in_tensor_map": [["I:out0", "add:in0"], ["I_1:out0", "add:in1"]],
# "src_acu_out_tensor_map": [["FakeQuantWithMinMaxVars:out0", "add:out0"]],
# "param_map": {"add": {}},
# "blob_map": {"add": {}},
# "acu_inter_flow": [],
# "priority_tip": 0,
# "pre_condition": None}
# ruler_list.append(r_add_quant)

r_greater = {
"ruler_name": "greater",
"src_ops_alias": ["Greater"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Greater:in0"], ["I_1:out0", "Greater:in1"]],
"src_out_tensor": ["Greater:out0"],
"acu_lys_alias": ["greater"],
"src_acu_in_tensor_map": [["I:out0", "greater:in0"], ["I_1:out0", "greater:in1"]],
"src_acu_out_tensor_map": [["Greater:out0", "greater:out0"]],
"param_map": {"greater": {}},
"blob_map": {"greater": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_greater)

r_greater_equal = {
"ruler_name": "greater_equal",
"src_ops_alias": ["GreaterEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GreaterEqual:in0"], ["I_1:out0", "GreaterEqual:in1"]],
"src_out_tensor": ["GreaterEqual:out0"],
"acu_lys_alias": ["greater_equal"],
"src_acu_in_tensor_map": [["I:out0", "greater_equal:in0"], ["I_1:out0", "greater_equal:in1"]],
"src_acu_out_tensor_map": [["GreaterEqual:out0", "greater_equal:out0"]],
"param_map": {"greater_equal": {}},
"blob_map": {"greater_equal": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_greater_equal)

r_less = {
"ruler_name": "less",
"src_ops_alias": ["Less"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Less:in0"], ["I_1:out0", "Less:in1"]],
"src_out_tensor": ["Less:out0"],
"acu_lys_alias": ["less"],
"src_acu_in_tensor_map": [["I:out0", "less:in0"], ["I_1:out0", "less:in1"]],
"src_acu_out_tensor_map": [["Less:out0", "less:out0"]],
"param_map": {"less": {}},
"blob_map": {"less": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_less)

r_less_equal = {
"ruler_name": "less_equal",
"src_ops_alias": ["LessEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LessEqual:in0"], ["I_1:out0", "LessEqual:in1"]],
"src_out_tensor": ["LessEqual:out0"],
"acu_lys_alias": ["less_equal"],
"src_acu_in_tensor_map": [["I:out0", "less_equal:in0"], ["I_1:out0", "less_equal:in1"]],
"src_acu_out_tensor_map": [["LessEqual:out0", "less_equal:out0"]],
"param_map": {"less_equal": {}},
"blob_map": {"less_equal": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_less_equal)

r_equal = {
"ruler_name": "equal",
"src_ops_alias": ["Equal"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Equal:in0"], ["I_1:out0", "Equal:in1"]],
"src_out_tensor": ["Equal:out0"],
"acu_lys_alias": ["equal"],
"src_acu_in_tensor_map": [["I:out0", "equal:in0"], ["I_1:out0", "equal:in1"]],
"src_acu_out_tensor_map": [["Equal:out0", "equal:out0"]],
"param_map": {"equal": {}},
"blob_map": {"equal": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_equal)

r_not_equal = {
"ruler_name": "not_equal",
"src_ops_alias": ["NotEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "NotEqual:in0"], ["I_1:out0", "NotEqual:in1"]],
"src_out_tensor": ["NotEqual:out0"],
"acu_lys_alias": ["not_equal"],
"src_acu_in_tensor_map": [["I:out0", "not_equal:in0"], ["I_1:out0", "not_equal:in1"]],
"src_acu_out_tensor_map": [["NotEqual:out0", "not_equal:out0"]],
"param_map": {"not_equal": {}},
"blob_map": {"not_equal": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_not_equal)

r_floor_div = {
"ruler_name": "floor_div",
"src_ops_alias": ["FloorDiv"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "FloorDiv:in0"], ["I_1:out0", "FloorDiv:in1"]],
"src_out_tensor": ["FloorDiv:out0"],
"acu_lys_alias": ["floor_div"],
"src_acu_in_tensor_map": [["I:out0", "floor_div:in0"], ["I_1:out0", "floor_div:in1"]],
"src_acu_out_tensor_map": [["FloorDiv:out0", "floor_div:out0"]],
"param_map": {"floor_div": {}},
"blob_map": {"floor_div": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_floor_div)

r_logical_or = {
"ruler_name": "logical_or",
"src_ops_alias": ["LogicalOr"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LogicalOr:in0"], ["I_1:out0", "LogicalOr:in1"]],
"src_out_tensor": ["LogicalOr:out0"],
"acu_lys_alias": ["logical_or"],
"src_acu_in_tensor_map": [["I:out0", "logical_or:in0"], ["I_1:out0", "logical_or:in1"]],
"src_acu_out_tensor_map": [["LogicalOr:out0", "logical_or:out0"]],
"param_map": {"logical_or": {}},
"blob_map": {"logical_or": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_logical_or)

r_logical_and = {
"ruler_name": "logical_and",
"src_ops_alias": ["LogicalAnd"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LogicalAnd:in0"], ["I_1:out0", "LogicalAnd:in1"]],
"src_out_tensor": ["LogicalAnd:out0"],
"acu_lys_alias": ["logical_and"],
"src_acu_in_tensor_map": [["I:out0", "logical_and:in0"], ["I_1:out0", "logical_and:in1"]],
"src_acu_out_tensor_map": [["LogicalAnd:out0", "logical_and:out0"]],
"param_map": {"logical_and": {}},
"blob_map": {"logical_and": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_logical_and)

r_pow = {
"ruler_name": "pow",
"src_ops_alias": ["Pow"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Pow:in0"], ["I_1:out0", "Pow:in1"]],
"src_out_tensor": ["Pow:out0"],
"acu_lys_alias": ["pow"],
"src_acu_in_tensor_map": [["I:out0", "pow:in0"], ["I_1:out0", "pow:in1"]],
"src_acu_out_tensor_map": [["Pow:out0", "pow:out0"]],
"param_map": {"pow": {}},
"blob_map": {"pow": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_pow)

r_real_div = {
"ruler_name": "real_div",
"src_ops_alias": ["RealDiv"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "RealDiv:in0"], ["I_1:out0", "RealDiv:in1"]],
"src_out_tensor": ["RealDiv:out0"],
"acu_lys_alias": ["real_div"],
"src_acu_in_tensor_map": [["I:out0", "real_div:in0"], ["I_1:out0", "real_div:in1"]],
"src_acu_out_tensor_map": [["RealDiv:out0", "real_div:out0"]],
"param_map": {"real_div": {}},
"blob_map": {"real_div": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_real_div)

r_select = {
"ruler_name": "select",
"src_ops_alias": ["Select"],
"src_inter_flow":[],
"src_in_anchor": [["I:out0", "Select:in0"], ["I_1:out0", "Select:in1"], ["I_2:out0", "Select:in2"]],
"src_out_tensor": ["Select:out0"],
"acu_lys_alias": ["where"],
"src_acu_in_tensor_map": [["I:out0", "where:in0"], ["I_1:out0", "where:in1"], ["I_2:out0", "where:in2"]],
"src_acu_out_tensor_map": [["Select:out0", "where:out0"]],
"acu_inter_flow": [],
"param_map": {"where": {}},
"blob_map": {"where": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_select)

r_where = {
"ruler_name": "where",
"src_ops_alias": ["Where"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Where:in0"]],
"src_out_tensor": ["Where:out0"],
"acu_lys_alias": ["where"],
"src_acu_in_tensor_map": [["I:out0", "where:in0"]],
"src_acu_out_tensor_map": [["Where:out0", "where:out0"]],
"acu_inter_flow": [],
"param_map": {"where": {}},
"blob_map": {"where": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_where)

r_square = {
"ruler_name": "square",
"src_ops_alias": ["Square"],
"src_inter_flow":[],
"src_in_anchor": [["I:out0", "Square:in0"]],
"src_out_tensor": ["Square:out0"],
"acu_lys_alias": ["square"],
"src_acu_in_tensor_map": [["I:out0", "square:in0"]],
"src_acu_out_tensor_map": [["Square:out0", "square:out0"]],
"acu_inter_flow": [],
"param_map": {"square": {}},
"blob_map": {"square": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_square)

r_sqrt = {
"ruler_name": "sqrt",
"src_ops_alias": ["Sqrt"],
"src_inter_flow":[],
"src_in_anchor": [["I:out0", "Sqrt:in0"]],
"src_out_tensor": ["Sqrt:out0"],
"acu_lys_alias": ["sqrt"],
"src_acu_in_tensor_map": [["I:out0", "sqrt:in0"]],
"src_acu_out_tensor_map": [["Sqrt:out0", "sqrt:out0"]],
"acu_inter_flow": [],
"param_map": {"sqrt": {}},
"blob_map": {"sqrt": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_sqrt)

r_floor = {
"ruler_name": "floor",
"src_ops_alias": ["Floor"],
"src_inter_flow":[],
"src_in_anchor": [["I:out0", "Floor:in0"]],
"src_out_tensor": ["Floor:out0"],
"acu_lys_alias": ["floor"],
"src_acu_in_tensor_map": [["I:out0", "floor:in0"]],
"src_acu_out_tensor_map": [["Floor:out0", "floor:out0"]],
"acu_inter_flow": [],
"param_map": {"floor": {}},
"blob_map": {"floor": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_floor)

r_rsqrt = {
"ruler_name": "rsqrt",
"src_ops_alias": ["Rsqrt"],
"src_inter_flow":[],
"src_in_anchor": [["I:out0", "Rsqrt:in0"]],
"src_out_tensor": ["Rsqrt:out0"],
"acu_lys_alias": ["rsqrt"],
"src_acu_in_tensor_map": [["I:out0", "rsqrt:in0"]],
"src_acu_out_tensor_map": [["Rsqrt:out0", "rsqrt:out0"]],
"acu_inter_flow": [],
"param_map": {"rsqrt": {}},
"blob_map": {"rsqrt": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_rsqrt)

r_exp = {
"ruler_name": "exp",
"src_ops_alias": ["Exp"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Exp:in0"]],
"src_out_tensor": ["Exp:out0"],
"acu_lys_alias": ["exp"],
"src_acu_in_tensor_map": [["I:out0", "exp:in0"]],
"src_acu_out_tensor_map": [["Exp:out0", "exp:out0"]],
"acu_inter_flow": [],
"param_map": {"exp": {}},
"blob_map": {"exp": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_exp)

r_subtract = {
"ruler_name": "subtract",
"src_ops_alias": ["Sub"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sub:in0"], ["I_1:out0", "Sub:in1"]],
"src_out_tensor": ["Sub:out0"],
"acu_lys_alias": ["subtract"],
"src_acu_in_tensor_map": [["I:out0", "subtract:in0"], ["I_1:out0", "subtract:in1"]],
"src_acu_out_tensor_map": [["Sub:out0", "subtract:out0"]],
"param_map": {"subtract": {}},
"blob_map": {"subtract": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_subtract)

r_add_2 = {
"ruler_name": "add_n_2_add",
"src_ops_alias": ["AddN"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AddN:in0"], ["I_1:out0", "AddN:in1"]],
"src_out_tensor": ["AddN:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map": [["I:out0", "add:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [["AddN:out0", "add:out0"]],
"param_map": {"add": {}},
"blob_map": {"add": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['AddN'], 'N', 0) == 2"}
ruler_list.append(r_add_2)

def r_add_n_template(in_tensor_count):
    src_acu_in_tensor_map = []
    for order in range(in_tensor_count - 1):
        src_acu_in_tensor_map.append(["I_{}:out0".format(order + 1), "add_{}:in1".format(order)])
    src_acu_in_tensor_map.append(["I_0:out0", "add_0:in0"])
    r_add_n_dict = {
        "ruler_name": "add_{}".format(in_tensor_count),
        "src_ops_alias": ["AddN"],
        "src_inter_flow": [],
        "src_in_anchor": [["I_{}:out0".format(order), "AddN:in{}".format(order)]
                          for order in range(in_tensor_count)],
        "src_out_tensor": ["AddN:out0"],
        "acu_lys_alias": ['add_{}'.format(order) for order in range(in_tensor_count - 1)],
        "src_acu_in_tensor_map": src_acu_in_tensor_map,
        "src_acu_out_tensor_map": [["AddN:out0", "add_{}:out0".format(in_tensor_count - 2)]],
        "param_map": {},
        "blob_map": {},
        "acu_inter_flow": [["add_{}:out0".format(order), "add_{}:in0".format(order + 1)]
                           for order in range(in_tensor_count - 2)],
        "priority_tip": 0,
        "pre_condition": "self.attr_pick(node['AddN'], 'N', 0) == {}".format(in_tensor_count)}
    return r_add_n_dict
#ruler_list.append(r_add_n_template(5))
#ruler_list.append(r_add_n_template(23))
#ruler_list.append(r_add_n_template(24))

def r_add_n_specail_template(addn_count, addn_input_list):
    src_acu_in_tensor_map = []

    for order in range(addn_count - 1):
        src_acu_in_tensor_map.append(["I_{}:out0".format(addn_input_list[order+1]), "add_{}:in1".format(order)])
    src_acu_in_tensor_map.append(["I_0:out0", "add_0:in0"])
    r_add_n_specail_dict = {
        "ruler_name": "add_specail_{}".format(addn_count),
        "src_ops_alias": ["AddN"],
        "src_inter_flow": [],
        "src_in_anchor": [["I_{}:out0".format(addn_input_list[order]), "AddN:in{}".format(order)]
                          for order in range(addn_count)],
        "src_out_tensor": ["AddN:out0"],
        "acu_lys_alias": ['add_{}'.format(order) for order in range(addn_count - 1)],
        "src_acu_in_tensor_map": src_acu_in_tensor_map,
        "src_acu_out_tensor_map": [["AddN:out0", "add_{}:out0".format(addn_count - 2)]],
        "param_map": {},
        "blob_map": {},
        "acu_inter_flow": [["add_{}:out0".format(order), "add_{}:in0".format(order + 1)]
                           for order in range(addn_count - 2)],
        "priority_tip": 0,
        "pre_condition": "self.attr_pick(node['AddN'], 'N', 0) == {}".format(addn_count)}
    return r_add_n_specail_dict

r_biasadd = {
"ruler_name": "biasadd",
"src_ops_alias": ["BiasAdd"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "BiasAdd:in0"], ["I_1:out0", "BiasAdd:in1"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map": [["I:out0", "add:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "add:out0"]],
"param_map": {"add": {}},
"blob_map": {"add": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_biasadd)

r_mul = {
"ruler_name": "multiply",
"src_ops_alias": ["Mul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mul:in0"], ["I_1:out0", "Mul:in1"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["multiply"],
"src_acu_in_tensor_map": [["I:out0", "multiply:in0"], ["I_1:out0", "multiply:in1"]],
"src_acu_out_tensor_map": [["Mul:out0", "multiply:out0"]],
"param_map": {"multiply": {}},
"blob_map": {"multiply": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mul)

#
# r_div = dict()
# r_div['src_ops_alias'] = ['Div_1']
# r_div['acu_lys_alias'] = ['divide_1']
# r_div['pre_condition'] = "not self.have_const_in_inputs(tensor['Div_1'])"
# r_div['src_acu_inputs_map'] =
# ['CODE', "[[['Div_1', 'in'+str(port)], ['divide_1', 'in'+str(port)]] for port in range(len(tensor['Div_1'].input))]"]
# ruler_list.append(r_div)
#
# r_abs = dict()
# r_abs['src_ops_alias'] = ['Abs_1']
# r_abs['acu_lys_alias'] = ['abs_1']
# ruler_list.append(r_abs)
#
# r_floor = dict()
# r_floor['src_ops_alias'] = ['Floor_1']
# r_floor['acu_lys_alias'] = ['floor_1']
# ruler_list.append(r_floor)
#
# r_neg = dict()
# r_neg['src_ops_alias'] = ['Neg']
# r_neg['acu_lys_alias'] = ['neg']
# ruler_list.append(r_neg)

r_abs = {
"ruler_name": "abs",
"src_ops_alias": ["Abs"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Abs:in0"]],
"src_out_tensor": ["Abs:out0"],
"acu_lys_alias": ["abs"],
"src_acu_in_tensor_map": [["I:out0", "abs:in0"]],
"src_acu_out_tensor_map": [["Abs:out0", "abs:out0"]],
"param_map": {"abs": {}},
"blob_map": {"abs": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_abs)

r_negative = {
"ruler_name": "neg",
"src_ops_alias": ["Neg"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Neg:in0"]],
"src_out_tensor": ["Neg:out0"],
"acu_lys_alias": ["neg"],
"src_acu_in_tensor_map": [["I:out0", "neg:in0"]],
"src_acu_out_tensor_map": [["Neg:out0", "neg:out0"]],
"param_map": {"neg": {}},
"blob_map": {"neg": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_negative)

r_clip_by_value = {
"ruler_name": "clipbyvalue",
"src_ops_alias": ["Maximum", "Minimum", "C", "C_1"],
"src_inter_flow": [["Minimum:out0", "Maximum:in0"], ["C:out0", "Maximum:in1"], ["C_1:out0", "Minimum:in1"]],
"src_in_anchor": [["I:out0", "Minimum:in0"]],
"src_out_tensor": ["Maximum:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Maximum:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {"clipbyvalue": {"clip_value_min": ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                              "clip_value_max": ['INT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],
                              }},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_clip_by_value)

r_maximum_2_elw = {
"ruler_name": "eltwise_max",
"src_ops_alias": ["Maximum"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Maximum:in0"], ["I_1:out0", "Maximum:in1"]],
"src_out_tensor": ["Maximum:out0"],
"acu_lys_alias": ["eltwise"],
"src_acu_in_tensor_map": [["I:out0", "eltwise:in0"], ["I_1:out0", "eltwise:in1"]],
"src_acu_out_tensor_map": [["Maximum:out0", "eltwise:out0"]],
"param_map": {"eltwise": {'operation': ['STRING', 'VALUE', "MAX"],}},
"blob_map": {"eltwise": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_maximum_2_elw)

r_argmax = {
"ruler_name": "argmax",
"src_ops_alias": ["ArgMax", "C"],
"src_inter_flow": [["C:out0", "ArgMax:in1"]],
"src_in_anchor": [["I:out0", "ArgMax:in0"]],
"src_out_tensor": ["ArgMax:out0"],
"acu_lys_alias": ["argmax"],
"src_acu_in_tensor_map": [["I:out0", "argmax:in0"]],
"src_acu_out_tensor_map": [["ArgMax:out0", "argmax:out0"]],
"param_map": {"argmax": {}},
"blob_map": {"argmax": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_argmax)

r_argmin={
"ruler_name": "argmin",
"src_ops_alias": ["ArgMin", "C"],
"src_inter_flow": [["C:out0", "ArgMin:in1"]],
"src_in_anchor": [["I:out0", "ArgMin:in0"]],
"src_out_tensor": ["ArgMin:out0"],
"acu_lys_alias": ["argmin"],
"src_acu_in_tensor_map": [["I:out0", "argmin:in0"]],
"src_acu_out_tensor_map": [["ArgMin:out0", "argmin:out0"]],
"acu_inter_flow": [],
"param_map": {"argmin": {'axis': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {"argmin": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_argmin)

r_placeholder = {
"ruler_name": "placeholder",
"src_ops_alias": ["Placeholder"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Placeholder:in0"]],
"src_out_tensor": ["Placeholder:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Placeholder:out0", "noop:out0"]],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_placeholder)

r_identity = {
"ruler_name": "identity",
"src_ops_alias": ["Identity"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Identity:in0"]],
"src_out_tensor": ["Identity:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Identity:out0", "noop:out0"]],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_identity)

r_cast = {
"ruler_name": "cast",
"src_ops_alias": ["Cast"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Cast:in0"]],
"src_out_tensor": ["Cast:out0"],
"acu_lys_alias": ["cast"],
"src_acu_in_tensor_map": [["I:out0", "cast:in0"]],
"src_acu_out_tensor_map": [["Cast:out0", "cast:out0"]],
"param_map": {"cast": {'in_data_type': ['STRING', 'CODE', "self.attr_pick(node['Cast'], 'SrcT')"],
                      'out_data_type': ['STRING', 'CODE',
                                        "self.tf_type_enum_to_ac_type(self.attr_pick(node['Cast'], 'DstT'))"]}},
"blob_map": {"cast": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_cast)

r_dropout_train_scale = {
"ruler_name": "dropout_train_scale",
"src_ops_alias": ["Mul", "RealDiv", "Floor", "C", "Add", "Add_1", "Mul_1", "C_1", "RandomUniform", "C_2", "C_3"],
"src_inter_flow": [["Floor:out0", "Mul:in1"], ["Add_1:out0", "Add:in1"], ["Add:out0", "Floor:in0"],
                   ["C_3:out0", "RandomUniform:in0"], ["RandomUniform:out0", "Mul_1:in0"], ["C:out0", "RealDiv:in1"],
                   ["C_1:out0", "Add_1:in1"], ["C_2:out0", "Mul_1:in1"], ["RealDiv:out0", "Mul:in0"],
                   ["Mul_1:out0", "Add_1:in0"], ["C:out0", "Add:in0"]],
"src_in_anchor": [["I:out0", "RealDiv:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "dropout:out0"]],
"param_map": {"dropout": {'scale_train':['BOOL', 'VALUE', True],
                          'ratio': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])"]}},
"blob_map": {"dropout": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_dropout_train_scale)


r_time_major_lstm_new = {
"ruler_name": "lstm_new",
"src_ops_alias": ["TensorArrayGatherV2", "TensorArrayV2", "Range", "Exit", "StridedSlice", "C", "TensorArraySizeV2",
"C_1", "Switch", "C_2", "C_3", "C_4", "C_5", "Merge", "LoopCond", "Enter", "NextIteration", "Less", "C_6",
"TensorArrayWriteV2", "Merge_1", "Enter_1", "Enter_2", "Identity", "Mul", "Identity_1", "Enter_3", "NextIteration_1",
"Switch_1", "Tanh", "Sigmoid", "C_7", "Add", "Add_1", "Split", "C_8", "Mul_1", "Mul_2", "C_9", "Add_2", "Identity_2",
"Sigmoid_1", "Sigmoid_2", "Tanh_1", "MatMul", "Enter_4", "Switch_2", "Add_3", "Concat", "Enter_5", "Identity_3",
"Merge_2", "C_10", "C_11", "TensorArrayReadV2", "Identity_4", "Identity_5", "C_12", "Enter_6", "NextIteration_2",
"Enter_7", "Enter_8", "Switch_3", "C_13", "Fill", "TensorArrayV2_1", "TensorArrayScatterV2", "Merge_3", "Pack",
"C_14", "Range_1", "C_15", "Enter_9", "NextIteration_3", "StridedSlice_1", "C_16", "C_17", "StridedSlice_2", "C_18",
"Fill_1", "C_19", "C_20", "C_21", "C_22", "C_23", "C_24", "C_25", "C_26", "Pack_1", "C_27", "C_28"],
"src_inter_flow": [["TensorArrayV2:out0", "TensorArrayGatherV2:in0"], ["Range:out0", "TensorArrayGatherV2:in1"],
["Exit:out0", "TensorArrayGatherV2:in2"], ["StridedSlice:out0", "TensorArrayV2:in0"], ["C:out0", "Range:in0"],
["TensorArraySizeV2:out0", "Range:in1"], ["C_1:out0", "Range:in2"], ["Switch:out0", "Exit:in0"],
["C_2:out0", "StridedSlice:in0"], ["C_3:out0", "StridedSlice:in1"], ["C_4:out0", "StridedSlice:in2"],
["C_5:out0", "StridedSlice:in3"], ["TensorArrayV2:out0", "TensorArraySizeV2:in0"],
["Exit:out0", "TensorArraySizeV2:in1"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"], ["Less:out0", "LoopCond:in0"],
["C_6:out0", "Enter:in0"], ["TensorArrayWriteV2:out0", "NextIteration:in0"], ["Merge_1:out0", "Less:in0"],
["Enter_1:out0", "Less:in1"], ["Enter_2:out0", "TensorArrayWriteV2:in0"],
["Identity:out0", "TensorArrayWriteV2:in1"], ["StridedSlice:out0", "Enter_1:in0"],
["Mul:out0", "TensorArrayWriteV2:in2"], ["Identity_1:out0", "TensorArrayWriteV2:in3"],
["TensorArrayV2:out0", "Enter_2:in0"], ["Enter_3:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
["Switch_1:out1", "Identity:in0"], ["Switch:out1", "Identity_1:in0"], ["Tanh:out0", "Mul:in0"],
["Sigmoid:out0", "Mul:in1"], ["C_7:out0", "Enter_3:in0"], ["LoopCond:out0", "Switch_1:in1"],
["Merge_1:out0", "Switch_1:in0"], ["Add:out0", "NextIteration_1:in0"], ["Add_1:out0", "Tanh:in0"],
["Split:out3", "Sigmoid:in0"], ["Identity:out0", "Add:in0"], ["C_8:out0", "Add:in1"],
["Mul_1:out0", "Add_1:in0"], ["Mul_2:out0", "Add_1:in1"], ["Identity:out4096", "C_8:in0"],
["C_9:out0", "Split:in0"], ["Add_2:out0", "Split:in1"], ["Identity_2:out0", "Mul_1:in0"],
["Sigmoid_1:out0", "Mul_1:in1"], ["Identity:out4096", "C_9:in0"], ["Sigmoid_2:out0", "Mul_2:in0"],
["Tanh_1:out0", "Mul_2:in1"], ["MatMul:out0", "Add_2:in0"], ["Enter_4:out0", "Add_2:in1"],
["Switch_2:out1", "Identity_2:in0"], ["Split:out0", "Sigmoid_2:in0"], ["Add_3:out0", "Sigmoid_1:in0"],
["Split:out1", "Tanh_1:in0"], ["LoopCond:out0", "Switch_2:in1"], ["Concat:out0", "MatMul:in0"],
["Enter_5:out0", "MatMul:in1"], ["Identity_3:out0", "Enter_4:in0"], ["Split:out2", "Add_3:in0"],
["Merge_2:out0", "Switch_2:in0"], ["C_10:out0", "Add_3:in1"], ["C_11:out0", "Concat:in0"],
["TensorArrayReadV2:out0", "Concat:in1"], ["Identity_4:out0", "Concat:in2"], ["Identity:out4096", "C_10:in0"],
["Identity_5:out0", "Enter_5:in0"], ["Identity:out4096", "C_11:in0"], ["C_12:out0", "Identity_3:in0"],
["Identity:out0", "TensorArrayReadV2:in1"], ["Enter_6:out0", "Merge_2:in0"],
["NextIteration_2:out0", "Merge_2:in1"], ["Enter_7:out0", "TensorArrayReadV2:in0"],
["Enter_8:out0", "TensorArrayReadV2:in2"], ["Switch_3:out1", "Identity_4:in0"],
["Add_1:out0", "NextIteration_2:in0"], ["C_13:out0", "Identity_5:in0"],
["LoopCond:out0", "Switch_3:in1"], ["Fill:out0", "Enter_6:in0"],
["StridedSlice:out0", "TensorArrayV2_1:in0"], ["TensorArrayV2_1:out0", "Enter_7:in0"],
["TensorArrayScatterV2:out0", "Enter_8:in0"], ["Merge_3:out0", "Switch_3:in0"],
["Pack:out0", "Fill:in0"], ["C_14:out0", "Fill:in1"], ["TensorArrayV2_1:out0", "TensorArrayScatterV2:in0"],
["Range_1:out0", "TensorArrayScatterV2:in1"], ["C_15:out0", "TensorArrayScatterV2:in3"],
["Enter_9:out0", "Merge_3:in0"], ["NextIteration_3:out0", "Merge_3:in1"], ["Mul:out0", "NextIteration_3:in0"],
["StridedSlice_1:out0", "Pack:in0"], ["C_16:out0", "Pack:in1"], ["C_17:out0", "Range_1:in0"],
["StridedSlice_2:out0", "Range_1:in1"], ["C_18:out0", "Range_1:in2"], ["Fill_1:out0", "Enter_9:in0"],
["C_19:out0", "StridedSlice_1:in0"], ["C_20:out0", "StridedSlice_1:in1"], ["C_21:out0", "StridedSlice_1:in2"],
["C_22:out0", "StridedSlice_1:in3"], ["C_23:out0", "StridedSlice_2:in0"], ["C_24:out0", "StridedSlice_2:in1"],
["C_25:out0", "StridedSlice_2:in2"], ["C_26:out0", "StridedSlice_2:in3"], ["Pack_1:out0", "Fill_1:in0"],
["C_27:out0", "Fill_1:in1"], ["StridedSlice_1:out0", "Pack_1:in0"], ["C_28:out0", "Pack_1:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV2:in2"]],
"src_out_tensor": ["TensorArrayGatherV2:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV2:out0", "lstm:out0"]],
"acu_inter_flow": [],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_10:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_13:out0'])[1] / 4"],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_13:out0'])"],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_12:out0'])"],
    }},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_time_major_lstm_new)

lstm_hidden_cell = {
"ruler_name": "lstm_hidden_cell",
"src_ops_alias": ["TensorArrayGatherV3", "Exit", "Exit_1", "TensorArrayV3", "Range", "Exit_2", "Switch",
    "Switch_1", "C", "C_1", "TensorArraySizeV3", "C_2", "Switch_2", "Merge", "LoopCond", "Merge_1",
    "Merge_2", "Enter", "NextIteration", "LogicalAnd", "Enter_1", "NextIteration_1", "Enter_2",
    "NextIteration_2", "Add", "Less", "Less_1", "Mul", "TensorArrayWriteV3", "Mul_1", "Mul_2",
    "Merge_3", "Enter_3", "Merge_4", "Enter_4", "Sigmoid", "Tanh", "Enter_5", "Identity", "Identity_1",
    "Sigmoid_1", "Identity_2", "Sigmoid_2", "Tanh_1", "Enter_6", "NextIteration_3", "Enter_7",
    "NextIteration_4", "C_3", "Split", "Switch_3", "Add_1", "C_4", "Add_2", "C_5", "Add_3", "C_6",
    "BiasAdd", "C_7", "Identity_3", "C_8", "C_9", "MatMul", "Enter_8", "Switch_4", "ConcatV2", "Enter_9",
    "C_10", "TensorArrayReadV3", "Identity_4", "C_11", "C_12", "Enter_10", "Enter_11", "TensorArrayV3_1",
    "TensorArrayScatterV3", "C_13"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayGatherV3:in0"],
    ["Range:out0", "TensorArrayGatherV3:in1"],
    ["Exit_2:out0", "TensorArrayGatherV3:in2"], ["Switch:out0", "Exit:in0"],
    ["Switch_1:out0", "Exit_1:in0"], ["C:out0", "TensorArrayV3:in0"], ["C_1:out0", "Range:in0"],
    ["TensorArraySizeV3:out0", "Range:in1"], ["C_2:out0", "Range:in2"],
    ["Switch_2:out0", "Exit_2:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Merge_1:out0", "Switch_1:in0"],
    ["TensorArrayV3:out0", "TensorArraySizeV3:in0"], ["Exit_2:out0", "TensorArraySizeV3:in1"],
    ["LoopCond:out0", "Switch_2:in1"], ["Merge_2:out0", "Switch_2:in0"], ["Enter:out0", "Merge:in0"],
    ["NextIteration:out0", "Merge:in1"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["Enter_1:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["Enter_2:out0", "Merge_2:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["Add:out0", "NextIteration:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["Less_1:out0", "LogicalAnd:in1"], ["TensorArrayV3:out1", "Enter_2:in0"],
    ["Mul:out0", "NextIteration_1:in0"], ["TensorArrayWriteV3:out0", "NextIteration_2:in0"],
    ["Mul_1:out0", "Add:in0"], ["Mul_2:out0", "Add:in1"], ["Merge_3:out0", "Less:in0"],
    ["Enter_3:out0", "Less:in1"], ["Merge_4:out0", "Less_1:in0"], ["Enter_4:out0", "Less_1:in1"],
    ["Sigmoid:out0", "Mul:in0"], ["Tanh:out0", "Mul:in1"], ["Mul:out0", "TensorArrayWriteV3:in2"],
    ["Enter_5:out0", "TensorArrayWriteV3:in0"], ["Identity:out0", "TensorArrayWriteV3:in1"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["C:out0", "Enter_3:in0"],
    ["Sigmoid_1:out0", "Mul_1:in0"], ["Identity_2:out0", "Mul_1:in1"], ["Sigmoid_2:out0", "Mul_2:in0"],
    ["Tanh_1:out0", "Mul_2:in1"], ["Enter_6:out0", "Merge_3:in0"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["TensorArrayV3:out0", "Enter_5:in0"],
    ["Enter_7:out0", "Merge_4:in0"], ["NextIteration_4:out0", "Merge_4:in1"], ["Add:out0", "Tanh:in0"],
    ["C_3:out0", "Enter_4:in0"], ["Split:out3", "Sigmoid:in0"], ["Switch_2:out1", "Identity_1:in0"],
    ["Switch:out1", "Identity_2:in0"], ["Switch_3:out1", "Identity:in0"],
    ["Add_1:out0", "Sigmoid_1:in0"], ["Split:out0", "Sigmoid_2:in0"], ["Split:out1", "Tanh_1:in0"],
    ["C_4:out0", "Enter_6:in0"], ["Add_2:out0", "NextIteration_3:in0"], ["C_5:out0", "Enter_7:in0"],
    ["LoopCond:out0", "Switch_3:in1"], ["Add_3:out0", "NextIteration_4:in0"],
    ["Merge_4:out0", "Switch_3:in0"], ["C_6:out0", "Split:in0"], ["BiasAdd:out0", "Split:in1"],
    ["Split:out2", "Add_1:in0"], ["C_7:out0", "Add_1:in1"], ["Identity:out0", "Add_3:in0"],
    ["Identity_3:out0", "Add_2:in0"], ["C_8:out0", "Add_2:in1"], ["C_9:out0", "Add_3:in1"],
    ["Identity_3:out4096", "C_6:in0"], ["MatMul:out0", "BiasAdd:in0"], ["Enter_8:out0", "BiasAdd:in1"],
    ["Identity_3:out4096", "C_7:in0"], ["LoopCond:out0", "Switch_4:in1"],
    ["Switch_4:out1", "Identity_3:in0"], ["Identity_3:out4096", "C_8:in0"],
    ["Identity_3:out4096", "C_9:in0"], ["Merge_3:out0", "Switch_4:in0"],
    ["ConcatV2:out0", "MatMul:in0"], ["Enter_9:out0", "MatMul:in1"],
    ["Switch_1:out1", "Identity_4:in0"], ["C_10:out0", "Enter_8:in0"],
    ["Identity:out0", "TensorArrayReadV3:in1"], ["TensorArrayReadV3:out0", "ConcatV2:in0"],
    ["Identity_4:out0", "ConcatV2:in1"], ["C_11:out0", "ConcatV2:in2"], ["C_12:out0", "Enter_9:in0"],
    ["C:out0", "TensorArrayV3_1:in0"], ["Enter_10:out0", "TensorArrayReadV3:in0"],
    ["Enter_11:out0", "TensorArrayReadV3:in2"], ["Identity_3:out4096", "C_11:in0"],
    ["TensorArrayV3_1:out0", "Enter_10:in0"], ["TensorArrayScatterV3:out0", "Enter_11:in0"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["C_13:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"], ["I_1:out0", "Enter_1:in0"],
    ["I_2:out0", "Enter:in0"]],
"src_out_tensor": ["TensorArrayGatherV3:out0", "Exit:out0", "Exit_1:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"], ["I_1:out0", "lstm:in1"], ["I_2:out0", "lstm:in2"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "lstm:out0"], ["Exit:out0", "lstm:out2"],
    ["Exit_1:out0", "lstm:out1"]],
"acu_inter_flow": [],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_7:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['Enter:out0'])[1]"],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_12:out0'])"],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_10:out0'])"],
    }},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(lstm_hidden_cell)

r_time_major_lstm = {
"ruler_name": "lstm",
"src_ops_alias":
["TensorArrayGatherV3", "TensorArrayV3", "Range", "Exit", "C", "C_1", "TensorArraySizeV3", "C_2", "Switch", "Merge",
"LoopCond", "Enter", "NextIteration", "Less", "TensorArrayWriteV3", "Merge_1", "Enter_1", "Enter_2", "Identity", "Mul",
"Identity_1", "Enter_3", "NextIteration_1", "Switch_1", "Sigmoid", "Tanh", "C_3", "Add", "Split", "Add_1", "C_4",
 "C_5","BiasAdd", "Mul_1", "Mul_2", "MatMul", "Enter_4", "Sigmoid_1", "Identity_2", "Sigmoid_2", "Tanh_1", "ConcatV2",
 "Enter_5",
"C_6", "Add_2", "Switch_2", "TensorArrayReadV3", "Identity_3", "C_7", "C_8", "C_9", "Merge_2", "Enter_6", "Enter_7",
 "Switch_3",
"Enter_8", "NextIteration_2", "TensorArrayV3_1", "TensorArrayScatterV3", "Merge_3", "C_10", "C_11", "Enter_9",
 "NextIteration_3", "C_12"],
"src_inter_flow": [["Split:out0", "Sigmoid_2:in0"], ["Split:out2", "Add_2:in0"], ["Switch_1:out1", "Identity:in0"],
                   ["Tanh:out0", "Mul:in1"],
    ["NextIteration_2:out0", "Merge_2:in1"], ["Mul_2:out0", "Add_1:in1"], ["Mul:out0", "NextIteration_3:in0"],
                   ["LoopCond:out0", "Switch_1:in1"],
    ["TensorArrayV3:out1", "Enter:in0"], ["TensorArrayV3:out0", "Enter_2:in0"],
                   ["Identity:out0", "TensorArrayReadV3:in1"], ["Tanh_1:out0", "Mul_2:in1"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["C_10:out0", "Enter_8:in0"],
                   ["TensorArrayScatterV3:out0", "Enter_7:in0"], ["Enter:out0", "Merge:in0"],
    ["Sigmoid_1:out0", "Mul_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"], ["Identity_3:out0", "ConcatV2:in1"],
                   ["LoopCond:out0", "Switch_3:in1"],
    ["Exit:out0", "TensorArraySizeV3:in1"], ["TensorArrayV3_1:out0", "Enter_6:in0"],
                   ["NextIteration:out0", "Merge:in1"], ["C:out0", "Enter_1:in0"],
    ["Identity:out4096", "C_5:in0"], ["ConcatV2:out0", "MatMul:in0"], ["C_4:out0", "Add:in1"],
                   ["Switch:out0", "Exit:in0"], ["Identity:out0", "Add:in0"],
    ["Enter_4:out0", "BiasAdd:in1"], ["C_1:out0", "Range:in0"], ["Switch_3:out1", "Identity_3:in0"],
                   ["C_9:out0", "Add_2:in1"], ["C_5:out0", "Split:in0"],
    ["MatMul:out0", "BiasAdd:in0"], ["Add_1:out0", "NextIteration_2:in0"], ["Enter_6:out0", "TensorArrayReadV3:in0"],
                   ["C:out0", "TensorArrayV3:in0"],
    ["C_6:out0", "Enter_4:in0"], ["Range:out0", "TensorArrayGatherV3:in1"], ["Merge_2:out0", "Switch_2:in0"],
                   ["Identity_2:out0", "Mul_1:in1"],
    ["TensorArrayV3:out0", "TensorArraySizeV3:in0"], ["C_11:out0", "TensorArrayScatterV3:in1"],
                   ["Merge:out0", "Switch:in0"], ["Switch_2:out1", "Identity_2:in0"],
    ["Enter_2:out0", "TensorArrayWriteV3:in0"], ["Split:out3", "Sigmoid:in0"],
                   ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["TensorArrayWriteV3:out0", "NextIteration:in0"],
    ["Identity:out4096", "C_7:in0"], ["Enter_5:out0", "MatMul:in1"], ["TensorArrayReadV3:out0", "ConcatV2:in0"],
                   ["C_7:out0", "ConcatV2:in2"], ["C_8:out0", "Enter_5:in0"],
    ["Sigmoid_2:out0", "Mul_2:in0"], ["C:out0", "TensorArrayV3_1:in0"], ["Mul_1:out0", "Add_1:in0"],
                   ["Enter_8:out0", "Merge_2:in0"], ["Enter_7:out0", "TensorArrayReadV3:in2"],
    ["Sigmoid:out0", "Mul:in0"], ["Mul:out0", "TensorArrayWriteV3:in2"], ["LoopCond:out0", "Switch:in1"],
                   ["TensorArraySizeV3:out0", "Range:in1"], ["Enter_1:out0", "Less:in1"],
    ["Less:out0", "LoopCond:in0"], ["Exit:out0", "TensorArrayGatherV3:in2"],
                   ["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["LoopCond:out0", "Switch_2:in1"],
    ["Split:out1", "Tanh_1:in0"], ["Identity:out4096", "C_4:in0"], ["C_2:out0", "Range:in2"],
                   ["Add_2:out0", "Sigmoid_1:in0"], ["Merge_3:out0", "Switch_3:in0"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["Switch:out1", "Identity_1:in0"],
                   ["Identity:out0", "TensorArrayWriteV3:in1"], ["Add:out0", "NextIteration_1:in0"],
    ["Enter_9:out0", "Merge_3:in0"], ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
                   ["Identity:out4096", "C_9:in0"], ["C_3:out0", "Enter_3:in0"],
    ["Merge_1:out0", "Switch_1:in0"], ["BiasAdd:out0", "Split:in1"], ["Merge_1:out0", "Less:in0"],
                   ["Enter_3:out0", "Merge_1:in0"], ["Add_1:out0", "Tanh:in0"], ["C_12:out0", "Enter_9:in0"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["TensorArrayGatherV3:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "lstm:out0"]],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['Enter_5:out0'])[1] / 4"],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_8:out0'])"],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_6:out0'])"],
    }},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_time_major_lstm)

r_fc_tensor_dot_bias_add_rule = {
    "ruler_name": "tensor_dot_bias_add",
    "src_ops_alias": ["BiasAdd", "Reshape", "C", "MatMul", "C_1", "Reshape_1", "C_2", "Transpose", "C_3", "C_4"],
    "src_inter_flow": [["C_4:out0", "Transpose:in1"], ["C_2:out0", "MatMul:in1"], ["Transpose:out0", "Reshape_1:in0"],
                       ["MatMul:out0", "Reshape:in0"],["Reshape:out0", "BiasAdd:in0"],
                       ["Reshape_1:out0", "MatMul:in0"],
                       ["C:out0", "BiasAdd:in1"], ["C_1:out0", "Reshape:in1"], ["C_3:out0", "Reshape_1:in1"]],
    "src_in_anchor": [["I:out0", "Transpose:in0"]],
    "src_out_tensor": ["BiasAdd:out0"],
    "acu_lys_alias": ["fullconnect"],
    "src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
    "src_acu_out_tensor_map": [["BiasAdd:out0", "fullconnect:out0"]],
    "param_map": {"fullconnect": {
        'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
        'bias': ['BOOL', 'PYFUNC', r_fc_tensor_dot_rule_get_param_bias()],
        'axis': ['INT', 'PYFUNC', r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_3:out0',
            transpose_out='Transpose:out0')],
        }},
    "blob_map": {"fullconnect": {
        'weight': ['PYFUNC', r_fc_tensor_dot_rule_get_weight()],
        'bias': ['PYFUNC', r_fc_tensor_dot_rule_get_bias()]
        }},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": r_fc_tensor_dot_rule_pre_condition()}
ruler_list.append(r_fc_tensor_dot_bias_add_rule)

r_fc_tensor_dot_rule = {
    "ruler_name": "tensor_dot",
    "src_ops_alias": ["Reshape", "MatMul", "C", "Reshape_1", "C_1", "Transpose", "C_2", "C_3"],
    "src_inter_flow": [["MatMul:out0", "Reshape:in0"], ["C:out0", "Reshape:in1"], ["Reshape_1:out0", "MatMul:in0"],
        ["C_1:out0", "MatMul:in1"], ["Transpose:out0", "Reshape_1:in0"], ["C_2:out0", "Reshape_1:in1"],
        ["C_3:out0", "Transpose:in1"]],
    "src_in_anchor": [["I:out0", "Transpose:in0"]],
    "src_out_tensor": ["Reshape:out0"],
    "acu_lys_alias": ["fullconnect"],
    "src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
    "src_acu_out_tensor_map": [["Reshape:out0", "fullconnect:out0"]],
    "acu_inter_flow": [],
    "param_map": {
        "fullconnect": {
            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
            'bias': ['BOOL', 'VALUE', False],
            'axis': ['INT', 'PYFUNC', r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_2:out0',
                transpose_out='Transpose:out0')],
        }
    },
    "blob_map": {
        "fullconnect": {
            'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"]
        }
    },
    "priority_tip": 0,
    "pre_condition": None }
ruler_list.append(r_fc_tensor_dot_rule)

#the r_mm_to_fc_rsp rule is for transformer ffn, the fc's input and output is 2D
r_mm_to_fc_rsp = {
    "ruler_name": "matmul_bias_add_to_fc_rsp",
    "src_ops_alias": ["BiasAdd", "Reshape", "C", "MatMul", "C_1", "Reshape_1", "C_2", "Transpose", "C_3", "C_4"],
    "src_inter_flow": [["C_4:out0", "Transpose:in1"], ["C_2:out0", "MatMul:in1"], ["Transpose:out0", "Reshape_1:in0"],
                       ["MatMul:out0", "Reshape:in0"],["Reshape:out0", "BiasAdd:in0"],
                       ["Reshape_1:out0", "MatMul:in0"],
                       ["C:out0", "BiasAdd:in1"], ["C_1:out0", "Reshape:in1"], ["C_3:out0", "Reshape_1:in1"]],
    "src_in_anchor": [["I:out0", "Transpose:in0"]],
    "src_out_tensor": ["BiasAdd:out0"],
    "acu_lys_alias": ["reshape", "fullconnect", "reshape_1"],
    "src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
    "src_acu_out_tensor_map": [["BiasAdd:out0", "reshape_1:out0"]],
    "param_map": {
        "reshape": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_3:out0'])"], },
        "fullconnect": {
            'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
            'bias': ['BOOL', 'PYFUNC', r_fc_tensor_dot_rule_get_param_bias()],
            'axis': ['INT', 'PYFUNC', r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_3:out0',
                                                                          transpose_out='Transpose:out0')],
            },
        "reshape_1": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_1:out0'])"], }},
    "blob_map": {"fullconnect": {
        'weight': ['PYFUNC', r_fc_tensor_dot_rule_get_weight()],
        'bias': ['PYFUNC', r_fc_tensor_dot_rule_get_bias()]
        }},
    "acu_inter_flow": [["reshape:out0", "fullconnect:in0"], ["fullconnect:out0", "reshape_1:in0"]],
    "priority_tip": 1,
    "pre_condition": "self.shape_pick(tensor['C_3:out0'])[0] == 2 and "\
                     "len(self.shape_pick(tensor['C:out0'])) < 2 and "\
                     "self.shape_pick(tensor['C_1:out0'])[0] > 2"
    }
#ruler_list.append(r_mm_to_fc_rsp) #replace it by FC2

r_older_tf_fullconnect_rules = {
"ruler_name": "fullconnect_per_title",
"src_ops_alias": ["BiasAdd", "MatMul", "C", "C_1","Identity_0" , "Identity_1","Reshape","C_2"],
"src_inter_flow": [["MatMul:out0", "BiasAdd:in0"], ["C:out0", "Identity_1:in0"], ["Identity_1:out0", "BiasAdd:in1"],
                   ["C_1:out0", "Identity_0:in0"], ["Identity_0:out0", "MatMul:in1"]
                  ,["Reshape:out0", "MatMul:in0"],["C_2:out0", "Reshape:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["BiasAdd:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["BiasAdd:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[-1]"],
                            'bias': ['BOOL', 'VALUE', True],
                            'axis': ['INT', 'PYFUNC', r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_2:out0',
                                                                                            transpose_out='I:out0')],
                              }},
"blob_map": {"fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'], trans=[1, 0] "\
                                                "if self.attr_pick(node['MatMul'], 'transpose_b', False) "\
                                                "else [0, 1])"],
                           'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],}},
"priority_tip": 0,
"pre_condition": r_fc_tensor_dot_rule_get_param_axis(reshape_param='C_2:out0', transpose_out='I:out0')}
# ruler_list.append(r_older_tf_fullconnect_rules)

@rule_pyfunc_def
def r_lstmunit_rule_get_param_weights(self, node, tensor, tensor_name): # 'C_2:out0'
    return self.shape_pick(tensor[tensor_name])[1] / 4

@rule_pyfunc_def
def r_lstmunit_rule_get_param_forget_bias(self, node, tensor, tensor_name): #
    forget_bias = self.tensor_to_numpy(tensor[tensor_name])
    return forget_bias

#[inputs + weights or num_proj, weights * 4], i, c, f, o
@rule_pyfunc_def
def r_lstmunit_rule_get_weight_w(self, node, tensor, weight_name, input_name, index):
    import numpy as np
    weight = self.tensor_to_numpy(tensor[weight_name])
    input_shape = self.shape_pick(tensor[input_name])
    input_weight = np.split(weight, [input_shape[-1]], axis=0)[0]
    splited_w = np.split(input_weight, 4, 1)[index]
    return splited_w

@rule_pyfunc_def
def r_lstmunit_rule_get_weight_h(self, node, tensor, weight_name, input_name, index):
    import numpy as np
    weight = self.tensor_to_numpy(tensor[weight_name])
    input_shape = self.shape_pick(tensor[input_name])
    output_weight = np.split(weight, [input_shape[-1]], axis=0)[1]
    splited_h = np.split(output_weight, 4, 1)[index]
    return splited_h

@rule_pyfunc_def
def r_lstmunit_rule_get_bias(self, node, tensor, bias_name, index):
    import numpy as np
    bias = self.tensor_to_numpy(tensor[bias_name])
    splited_bias = np.split(bias, 4, 0)[index]
    return splited_bias

r_lstmunit_rule = {
    "ruler_name": "lstmunit",
    "src_ops_alias": ["Mul", "Tanh", "Sigmoid", "Add", "Split", "Mul_1", "Mul_2", "C", "BiasAdd", "Sigmoid_1",
                      "Sigmoid_2", "Tanh_1", "MatMul", "C_1", "Add_1", "ConcatV2", "C_2", "C_3", "C_4"],
    "src_inter_flow": [["Split:out3", "Sigmoid:in0"], ["MatMul:out0", "BiasAdd:in0"], ["Mul_1:out0", "Add:in0"],
                       ["Sigmoid_1:out0", "Mul_1:in1"],
["Split:out2", "Add_1:in0"], ["Split:out0", "Sigmoid_2:in0"], ["C:out0", "Split:in0"], ["Split:out1", "Tanh_1:in0"],
["ConcatV2:out0", "MatMul:in0"],
["Mul_2:out0", "Add:in1"], ["Tanh_1:out0", "Mul_2:in1"], ["C_4:out0", "ConcatV2:in2"], ["Add_1:out0", "Sigmoid_1:in0"],
                       ["Sigmoid:out0", "Mul:in1"],
["C_2:out0", "MatMul:in1"], ["BiasAdd:out0", "Split:in1"], ["C_3:out0", "Add_1:in1"], ["C_1:out0", "BiasAdd:in1"],
                       ["Sigmoid_2:out0", "Mul_2:in0"],
["Add:out0", "Tanh:in0"], ["Tanh:out0", "Mul:in0"]],
    "src_in_anchor": [["I:out0", "ConcatV2:in0"], ["I_1:out0", "ConcatV2:in1"], ["I_2:out0", "Mul_1:in0"]],
    "src_out_tensor": ["Mul:out0", "Add:out0"],
    "acu_lys_alias": ["lstmunit"],
    "src_acu_in_tensor_map": [["I:out0", "lstmunit:in0"], ["I_1:out0", "lstmunit:in1"], ["I_2:out0", "lstmunit:in2"]],
    "src_acu_out_tensor_map": [["Mul:out0", "lstmunit:out0"], ["Mul:out0", "lstmunit:out1"],
                               ["Add:out0", "lstmunit:out2"]],
    "param_map": {"lstmunit": {
        'weights': ['INT', 'PYFUNC', r_lstmunit_rule_get_param_weights(tensor_name='C_2:out0')],
        'num_proj': ['ORIGIN', 'VALUE', None],
        'forget_bias': ['FLOAT', 'PYFUNC', r_lstmunit_rule_get_param_forget_bias(tensor_name='C_3:out0')],
        }},
    "blob_map": {"lstmunit": {
        'wi': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_2:out0', input_name='I:out0', index=0)],
        'wc': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_2:out0', input_name='I:out0', index=1)],
        'wf': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_2:out0', input_name='I:out0', index=2)],
        'wo': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_2:out0', input_name='I:out0', index=3)],
        'hi': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_2:out0', input_name='I:out0', index=0)],
        'hc': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_2:out0', input_name='I:out0', index=1)],
        'hf': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_2:out0', input_name='I:out0', index=2)],
        'ho': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_2:out0', input_name='I:out0', index=3)],
        'bi': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=0)],
        'bc': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=1)],
        'bf': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=2)],
        'bo': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=3)],
        }},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_lstmunit_rule)

r_lstmunit_rule_2_0_0 = {
"ruler_name": "lstmunit_x",
"src_ops_alias": ["Slice", "Slice_1", "Pack", "C", "C_1", "Pack_1", "C_2", "C_3", "AddV2", "Mul", "Mul_1", "Mul_2",
    "Sigmoid", "Tanh", "Sigmoid_1", "Sigmoid_2", "Tanh_1", "Split", "AddV2_1", "C_4", "BiasAdd", "C_5",
    "MatMul", "C_6", "ConcatV2", "C_7", "C_8"],
"src_inter_flow": [["Pack:out0", "Slice:in0"], ["C:out0", "Slice:in1"], ["C_1:out0", "Slice:in2"],
    ["Pack_1:out0", "Slice_1:in0"], ["C_2:out0", "Slice_1:in1"], ["C_3:out0", "Slice_1:in2"],
    ["AddV2:out0", "Pack:in0"], ["Mul:out0", "Pack:in1"], ["AddV2:out0", "Pack_1:in0"],
    ["Mul:out0", "Pack_1:in1"], ["Mul_1:out0", "AddV2:in0"], ["Mul_2:out0", "AddV2:in1"],
    ["Sigmoid:out0", "Mul:in0"], ["Tanh:out0", "Mul:in1"], ["Sigmoid_1:out0", "Mul_1:in0"],
    ["Sigmoid_2:out0", "Mul_2:in0"], ["Tanh_1:out0", "Mul_2:in1"], ["Split:out3", "Sigmoid:in0"],
    ["AddV2:out0", "Tanh:in0"], ["AddV2_1:out0", "Sigmoid_1:in0"], ["Split:out0", "Sigmoid_2:in0"],
    ["Split:out1", "Tanh_1:in0"], ["C_4:out0", "Split:in0"], ["BiasAdd:out0", "Split:in1"],
    ["Split:out2", "AddV2_1:in0"], ["C_5:out0", "AddV2_1:in1"], ["MatMul:out0", "BiasAdd:in0"],
    ["C_6:out0", "BiasAdd:in1"], ["ConcatV2:out0", "MatMul:in0"], ["C_7:out0", "MatMul:in1"],
    ["C_8:out0", "ConcatV2:in2"]],
"src_in_anchor": [["I:out0", "ConcatV2:in0"], ["I_1:out0", "ConcatV2:in1"], ["I_2:out0", "Mul_1:in1"]],
"src_out_tensor": ["Slice:out0", "Slice_1:out0"],
"acu_lys_alias": ["lstmunit"],
"src_acu_in_tensor_map": [["I:out0", "lstmunit:in0"], ["I_1:out0", "lstmunit:in1"], ["I_2:out0", "lstmunit:in2"]],
"src_acu_out_tensor_map": [["Slice_1:out0", "lstmunit:out0"], ["Slice_1:out0", "lstmunit:out1"],
        ["Slice:out0", "lstmunit:out2"]],
"acu_inter_flow": [],
"param_map": {"lstmunit": {
    'weights': ['INT', 'PYFUNC', r_lstmunit_rule_get_param_weights(tensor_name='C_7:out0')],
    'num_proj': ['ORIGIN', 'VALUE', None],
    'forget_bias': ['FLOAT', 'PYFUNC', r_lstmunit_rule_get_param_forget_bias(tensor_name='C_5:out0')],
    }},
"blob_map": {"lstmunit": {
    'wi': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_7:out0', input_name='I:out0', index=0)],
    'wc': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_7:out0', input_name='I:out0', index=1)],
    'wf': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_7:out0', input_name='I:out0', index=2)],
    'wo': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_7:out0', input_name='I:out0', index=3)],
    'hi': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_7:out0', input_name='I:out0', index=0)],
    'hc': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_7:out0', input_name='I:out0', index=1)],
    'hf': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_7:out0', input_name='I:out0', index=2)],
    'ho': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_7:out0', input_name='I:out0', index=3)],
    'bi': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_6:out0', index=0)],
    'bc': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_6:out0', index=1)],
    'bf': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_6:out0', index=2)],
    'bo': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_6:out0', index=3)],
    }},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_lstmunit_rule_2_0_0)

r_static_lstmunit = {
    "ruler_name": "static_lstmunit",
    "src_ops_alias": ["Add", "Mul", "Mul_1", "Mul_2", "Sigmoid", "Tanh", "Sigmoid_1", "Sigmoid_2", "Tanh_1",
        "Split", "Add_1", "C", "BiasAdd", "C_1", "MatMul", "C_2", "ConcatV2", "C_3", "C_4"],
    "src_inter_flow": [["Mul_1:out0", "Add:in0"], ["Mul_2:out0", "Add:in1"], ["Sigmoid:out0", "Mul:in0"],
        ["Tanh:out0", "Mul:in1"], ["Sigmoid_1:out0", "Mul_1:in0"], ["Sigmoid_2:out0", "Mul_2:in0"],
        ["Tanh_1:out0", "Mul_2:in1"], ["Split:out3", "Sigmoid:in0"], ["Add:out0", "Tanh:in0"],
        ["Add_1:out0", "Sigmoid_1:in0"], ["Split:out0", "Sigmoid_2:in0"], ["Split:out1", "Tanh_1:in0"],
        ["C:out0", "Split:in0"], ["BiasAdd:out0", "Split:in1"], ["Split:out2", "Add_1:in0"],
        ["C_1:out0", "Add_1:in1"], ["MatMul:out0", "BiasAdd:in0"], ["C_2:out0", "BiasAdd:in1"],
        ["ConcatV2:out0", "MatMul:in0"], ["C_3:out0", "MatMul:in1"], ["C_4:out0", "ConcatV2:in2"]],
    "src_in_anchor": [["I:out0", "ConcatV2:in0"], ["I_1:out0", "ConcatV2:in1"], ["I_2:out0", "Mul_1:in1"]],
    "src_out_tensor": ["Add:out0", "Mul:out0"],
    "acu_lys_alias": ["lstmunit"],
    "src_acu_in_tensor_map": [["I:out0", "lstmunit:in0"], ["I_1:out0", "lstmunit:in1"], ["I_2:out0", "lstmunit:in2"]],
    "src_acu_out_tensor_map": [["Mul:out0", "lstmunit:out0"], ["Mul:out0", "lstmunit:out1"],
                               ["Add:out0", "lstmunit:out2"]],
    "acu_inter_flow": [],
    "param_map": {"lstmunit": {
        'weights': ['INT', 'PYFUNC', r_lstmunit_rule_get_param_weights(tensor_name='C_3:out0')],
        'num_proj': ['ORIGIN', 'VALUE', None],
        'forget_bias': ['FLOAT', 'PYFUNC', r_lstmunit_rule_get_param_forget_bias(tensor_name='C_1:out0')],
        }},
    "blob_map": {"lstmunit": {
        'wi': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=0)],
        'wc': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=1)],
        'wf': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=2)],
        'wo': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=3)],
        'hi': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=0)],
        'hc': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=1)],
        'hf': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=2)],
        'ho': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=3)],
        'bi': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_2:out0', index=0)],
        'bc': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_2:out0', index=1)],
        'bf': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_2:out0', index=2)],
        'bo': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_2:out0', index=3)],
    }},
    "priority_tip": 0,
    "pre_condition": "len(self.shape_pick(tensor['C_2:out0'])) < 2 and "\
        "self.attr_pick(node['MatMul'], 'transpose_a', False) == False"}
ruler_list.append(r_static_lstmunit)

r_lstmunit_rule_2_0_0_x = {
    "ruler_name": "r_lstmunit_rule_2_0_0_x",
    "src_ops_alias": ["Mul", "AddV2", "Sigmoid", "Tanh", "Mul_1", "Mul_2", "Split", "Sigmoid_1", "Sigmoid_2", "Tanh_1",
        "C", "BiasAdd", "AddV2_1", "MatMul", "C_1", "C_2", "ConcatV2", "C_3", "C_4"],
    "src_inter_flow": [["Sigmoid:out0", "Mul:in0"], ["Tanh:out0", "Mul:in1"], ["Mul_1:out0", "AddV2:in0"],
        ["Mul_2:out0", "AddV2:in1"], ["Split:out3", "Sigmoid:in0"], ["AddV2:out0", "Tanh:in0"],
        ["Sigmoid_1:out0", "Mul_1:in0"], ["Sigmoid_2:out0", "Mul_2:in0"], ["Tanh_1:out0", "Mul_2:in1"],
        ["C:out0", "Split:in0"], ["BiasAdd:out0", "Split:in1"], ["AddV2_1:out0", "Sigmoid_1:in0"],
        ["Split:out0", "Sigmoid_2:in0"], ["Split:out1", "Tanh_1:in0"], ["MatMul:out0", "BiasAdd:in0"],
        ["C_1:out0", "BiasAdd:in1"], ["Split:out2", "AddV2_1:in0"], ["C_2:out0", "AddV2_1:in1"],
        ["ConcatV2:out0", "MatMul:in0"], ["C_3:out0", "MatMul:in1"], ["C_4:out0", "ConcatV2:in2"]],
    "src_in_anchor": [["I:out0", "ConcatV2:in0"], ["I_1:out0", "ConcatV2:in1"], ["I_2:out0", "Mul_1:in1"]],
    "src_out_tensor": ["Mul:out0", "Mul:out0", "AddV2:out0"],
    "acu_lys_alias": ["lstmunit"],
    "src_acu_in_tensor_map": [["I:out0", "lstmunit:in0"], ["I_1:out0", "lstmunit:in1"], ["I_2:out0", "lstmunit:in2"]],
    "src_acu_out_tensor_map": [
        ["Mul:out0", "lstmunit:out0"],
        ["Mul:out0", "lstmunit:out1"],
        ["AddV2:out0", "lstmunit:out2"]
    ],
    "acu_inter_flow": [],
    "param_map": {"lstmunit": {
        'weights': ['INT', 'PYFUNC', r_lstmunit_rule_get_param_weights(tensor_name='C_3:out0')],
        'num_proj': ['ORIGIN', 'VALUE', None],
        'forget_bias': ['FLOAT', 'PYFUNC', r_lstmunit_rule_get_param_forget_bias(tensor_name='C_2:out0')],
     }},
    "blob_map": {"lstmunit": {
        'wi': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=0)],
        'wc': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=1)],
        'wf': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=2)],
        'wo': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_3:out0', input_name='I:out0', index=3)],
        'hi': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=0)],
        'hc': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=1)],
        'hf': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=2)],
        'ho': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_3:out0', input_name='I:out0', index=3)],
        'bi': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=0)],
        'bc': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=1)],
        'bf': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=2)],
        'bo': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_1:out0', index=3)],
     }},
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_lstmunit_rule_2_0_0_x)

@rule_pyfunc_def
def r_stack_concat_get_shape(self, node, tensor, concat_out):
    shape = self.shape_pick(tensor[concat_out])
    return list(shape)

@rule_pyfunc_def
def r_stack_concat_get_axis(self, node, tensor, axis_tensor_name):
    axis = self.tensor_to_numpy(tensor[axis_tensor_name])
    return axis

@rule_pyfunc_def
def r_stack_concat_gen_vari(self, node, tensor, concat_out):
    import numpy as np
    shape = self.shape_pick(tensor[concat_out])
    data = np.zeros(shape, dtype=np.float32)
    return data

r_stack_concat = {
"ruler_name": "stack_concat",
"src_ops_alias": ["ConcatV2", "NextIteration", "C"],
"src_inter_flow": [["C:out0", "ConcatV2:in2"], ["ConcatV2:out0", "NextIteration:in0"]],
"src_in_anchor": [["I:out0", "ConcatV2:in1"], ["I_1:out0", "ConcatV2:in0"]],
"src_out_tensor": ["ConcatV2:out0", "NextIteration:out0"],
"acu_lys_alias": ["stack_concat", "input", "variable"],
"src_acu_in_tensor_map": [["I:out0", "stack_concat:in0"]],
"src_acu_out_tensor_map": [["ConcatV2:out0", "stack_concat:out0"], ["NextIteration:out0", "stack_concat:out0"]],
"acu_inter_flow": [["input:out0", "stack_concat:in1"], ["variable:out0", "stack_concat:in2"]],
"param_map": {"stack_concat": {'axis': ['INT', 'PYFUNC', r_stack_concat_get_axis(axis_tensor_name='C:out0')],
                               'shape': ['ORIGIN', 'PYFUNC', r_stack_concat_get_shape(concat_out='ConcatV2:out0')],
                               },
              "input": {'shape': ['ORIGIN', 'VALUE', [1]]},
              "variable": {'shape': ['ORIGIN', 'PYFUNC', r_stack_concat_get_shape(concat_out='ConcatV2:out0')]},},
"blob_map": {"variable": {'data': ['PYFUNC', r_stack_concat_gen_vari(concat_out='ConcatV2:out0')],}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_stack_concat)

r_multi_attention ={
"ruler_name": "multi_attention",
"src_ops_alias": ["Reshape", "MatMul", "C", "Reshape_1", "C_1", "Transpose", "C_2",
    "Reshape_2", "C_3", "Transpose_1", "C_4", "BatchMatMul", "C_5", "Reshape_3",
    "Transpose_2", "Softmax", "C_6", "Reshape_4", "C_7", "Reshape_5", "ConcatV2",
    "C_8", "Add", "C_9", "Reshape_6", "C_10", "BatchMatMul_1", "MatMul_1", "C_11",
    "Mul", "Transpose_3", "Reshape_7", "C_12", "Transpose_4", "C_13", "Reshape_8",
    "C_14", "Transpose_5", "C_15", "Reshape_9", "C_16", "ConcatV2_1", "C_17", "C_18",
    "Reshape_10", "C_19", "Reshape_11", "C_20", "MatMul_2", "C_21", "MatMul_3", "C_22",
    "Reshape_12", "C_23", "Reshape_13", "C_24", "Transpose_6", "C_25", "Transpose_7",
    "C_26", "C_27", "C_28"],
"src_inter_flow": [["MatMul:out0", "Reshape:in0"], ["C:out0", "Reshape:in1"],
    ["Reshape_1:out0", "MatMul:in0"], ["C_1:out0", "MatMul:in1"],
    ["Transpose:out0", "Reshape_1:in0"], ["C_2:out0", "Reshape_1:in1"],
    ["Reshape_2:out0", "Transpose:in0"], ["C_3:out0", "Transpose:in1"],
    ["Transpose_1:out0", "Reshape_2:in0"], ["C_4:out0", "Reshape_2:in1"],
    ["BatchMatMul:out0", "Transpose_1:in0"], ["C_5:out0", "Transpose_1:in1"],
    ["Reshape_3:out0", "BatchMatMul:in0"], ["Transpose_2:out0", "BatchMatMul:in1"],
    ["Softmax:out0", "Reshape_3:in0"], ["C_6:out0", "Reshape_3:in1"],
    ["Reshape_4:out0", "Transpose_2:in0"], ["C_7:out0", "Transpose_2:in1"],
    ["Reshape_5:out0", "Softmax:in0"], ["ConcatV2:out0", "Reshape_4:in0"],
    ["C_8:out0", "Reshape_4:in1"], ["Add:out0", "Reshape_5:in0"],
    ["C_9:out0", "Reshape_5:in1"], ["Reshape_6:out0", "ConcatV2:in1"],
    ["C_10:out0", "ConcatV2:in2"], ["BatchMatMul_1:out0", "Add:in0"],
    ["MatMul_1:out0", "Reshape_6:in0"], ["C_11:out0", "Reshape_6:in1"],
    ["Mul:out0", "BatchMatMul_1:in0"], ["Transpose_3:out0", "BatchMatMul_1:in1"],
    ["Reshape_7:out0", "MatMul_1:in0"], ["C_12:out0", "MatMul_1:in1"],
    ["Transpose_4:out0", "Mul:in0"], ["C_13:out0", "Mul:in1"],
    ["Reshape_8:out0", "Transpose_3:in0"], ["C_14:out0", "Transpose_3:in1"],
    ["Transpose_5:out0", "Reshape_7:in0"], ["C_15:out0", "Reshape_7:in1"],
    ["Reshape_9:out0", "Transpose_4:in0"], ["C_16:out0", "Transpose_4:in1"],
    ["ConcatV2_1:out0", "Reshape_8:in0"], ["C_17:out0", "Reshape_8:in1"],
    ["C_18:out0", "Transpose_5:in1"], ["Reshape_10:out0", "Reshape_9:in0"],
    ["C_19:out0", "Reshape_9:in1"], ["Reshape_11:out0", "ConcatV2_1:in1"],
    ["C_20:out0", "ConcatV2_1:in2"], ["MatMul_2:out0", "Reshape_10:in0"],
    ["C_21:out0", "Reshape_10:in1"], ["MatMul_3:out0", "Reshape_11:in0"],
    ["C_22:out0", "Reshape_11:in1"], ["Reshape_12:out0", "MatMul_2:in0"],
    ["C_23:out0", "MatMul_2:in1"], ["Reshape_13:out0", "MatMul_3:in0"],
    ["C_24:out0", "MatMul_3:in1"], ["Transpose_6:out0", "Reshape_12:in0"],
    ["C_25:out0", "Reshape_12:in1"], ["Transpose_7:out0", "Reshape_13:in0"],
    ["C_26:out0", "Reshape_13:in1"], ["C_27:out0", "Transpose_6:in1"],
    ["C_28:out0", "Transpose_7:in1"]],
"src_in_anchor": [["I:out0", "Transpose_5:in0"], ["I:out0", "Transpose_7:in0"],
    ["I:out0", "Transpose_6:in0"], ["I_1:out0", "Add:in1"],
    ["I_2:out0", "ConcatV2_1:in0"], ["I_3:out0", "ConcatV2:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["variable","input","input_1", "reshape_11", "fullconnect_3",
    "reshape_10", "stack_concat_1", "reshape_9", "permute_3", "reshape_8",
    "fullconnect_2", "reshape_7", "stack_concat", "reshape_6", "permute_2",
    "reshape_5", "fullconnect_1", "reshape_4", "permute_1", "multiply",
    "matmul_1", "add", "reshape_3", "softmax", "reshape_2", "matmul", "permute",
    "reshape_1", "fullconnect", "reshape", "variable_1", "variable_2"],
"src_acu_in_tensor_map": [["I:out0", "reshape_11:in0"], ["I:out0", "reshape_8:in0"],
    ["I:out0", "reshape_5:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"acu_inter_flow": [
    ["variable:out0", "multiply:in1"], ["reshape_11:out0", "fullconnect_3:in0"],
    ["fullconnect_3:out0", "reshape_10:in0"], ["reshape_10:out0", "stack_concat_1:in0"],
    ["input_1:out0", "stack_concat_1:in1"], ["stack_concat_1:out0", "reshape_9:in0"],
    ["reshape_9:out0", "permute_3:in0"], ["permute_3:out0", "matmul_1:in1"],
    ["reshape_8:out0", "fullconnect_2:in0"], ["fullconnect_2:out0", "reshape_7:in0"],
    ["reshape_7:out0", "stack_concat:in0"], ["input:out0", "stack_concat:in1"],
    ["stack_concat:out0", "reshape_6:in0"], ["reshape_6:out0", "permute_2:in0"],
    ["permute_2:out0", "matmul:in1"],["reshape_5:out0", "fullconnect_1:in0"],
    ["fullconnect_1:out0", "reshape_4:in0"], ["reshape_4:out0", "permute_1:in0"],
    ["permute_1:out0", "multiply:in0"], ["multiply:out0", "matmul_1:in0"],
    ["matmul_1:out0", "add:in0"],["add:out0", "reshape_3:in0"],
    ["reshape_3:out0", "softmax:in0"], ["softmax:out0", "reshape_2:in0"],
    ["reshape_2:out0", "matmul:in0"], ["matmul:out0", "permute:in0"],
    ["permute:out0", "reshape_1:in0"], ["reshape_1:out0", "fullconnect:in0"],
    ["fullconnect:out0", "reshape:in0"], ["variable_2:out0", "stack_concat_1:in2"],
    ["variable_1:out0", "stack_concat:in2"]],
"param_map": {
    "variable": {'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['C_13:out0'])"]},
    "variable_1": {'shape': ['ORIGIN', 'VALUE', [1,32,256]]},
    "variable_2": {'shape': ['ORIGIN', 'VALUE', [1,32,256]]},
    "fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_1:out0'])[1]"],
        'bias': ['BOOL', 'VALUE', False]},
    "fullconnect_1": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_23:out0'])[1]"],
        'bias': ['BOOL', 'VALUE', False]},
    "fullconnect_2": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_12:out0'])[1]"],
        'bias': ['BOOL', 'VALUE', False]},
    "fullconnect_3": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_24:out0'])[1]"],
        'bias': ['BOOL', 'VALUE', False]},
    "input": {'shape': ['ORIGIN', 'VALUE', [1]]},
    "input_1": {'shape': ['ORIGIN', 'VALUE', [1]]},
    "stack_concat": {},
    "stack_concat_1": {},
    "permute": {'perm': ['STRING', 'CODE',
        "' '.join([str(perm) for perm in self.tensor_to_numpy(tensor['C_5:out0'])])"],},
    "permute_1": {'perm': ['STRING', 'CODE',
        "' '.join([str(perm) for perm in self.tensor_to_numpy(tensor['C_16:out0'])])"],},
    "permute_2": {'perm': ['STRING', 'CODE',
        "' '.join([str(perm) for perm in self.tensor_to_numpy(tensor['C_7:out0'])])"],},
    "permute_3": {'perm': ['STRING', 'CODE',
        "' '.join([str(perm) for perm in self.tensor_to_numpy(tensor['C_14:out0'])])"],},
    "reshape": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C:out0'])"],},
    "reshape_1": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_2:out0'])"],},
    "reshape_2": {'shape': ['INTS', 'VALUE', [1,4,1,32]],},
    "reshape_3": {'shape': ['INTS', 'VALUE', [-1,32]],},
    "reshape_4": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_19:out0'])"],},
    "reshape_5": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_25:out0'])"],},
    "reshape_6": {'shape': ['INTS', 'VALUE', [1,32,4,64]],},
    "reshape_7": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_11:out0'])"],},
    "reshape_8": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_15:out0'])"],},
    "reshape_9": {'shape': ['INTS', 'VALUE', [1,32,4,64]],},
    "reshape_10": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_22:out0'])"],},
    "reshape_11": {'shape': ['INTS', 'CODE', "self.reshape_shape(tensor['C_26:out0'])"],},
    "matmul": {'transpose_a': ['INT', 'CODE', "self.attr_pick(node['BatchMatMul'], 'adj_x', False)"],
        'transpose_b': ['INT', 'CODE', "self.attr_pick(node['BatchMatMul'], 'adj_y', False)"],},
    "matmul_1": {'transpose_a': ['INT', 'CODE', "self.attr_pick(node['BatchMatMul_1'], 'adj_x', False)"],
        'transpose_b': ['INT', 'CODE', "self.attr_pick(node['BatchMatMul_1'], 'adj_y', False)"],},
    "multiply": {},
    "softmax": {},
    "add": {}},
"blob_map": {
    "variable": {'data': ['CODE', "np.array([self.tensor_to_numpy(tensor['C_13:out0'])], dtype=np.float32) "\
        "if self.tensor_to_numpy(tensor['C_13:out0']).shape == () "\
        "else self.tensor_to_numpy(tensor['C_13:out0'], dtype='float32')"],},
    "variable_1": {'data': ['CODE', "np.zeros((1,32,256), dtype=np.float32)"],},
    "variable_2": {'data': ['CODE', "np.zeros((1,32,256), dtype=np.float32)"],},
    "fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"], },
    "fullconnect_1": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_23:out0'])"], },
    "fullconnect_2": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_12:out0'])"],},
    "fullconnect_3": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_24:out0'])"],},
    "reshape_8": {},
    "stack_concat_1": {},
    "reshape_7": {},
    "reshape": {},
    "permute_2": {},
    "permute": {},
    "permute_3": {},
    "reshape_1": {},
    "reshape_2": {},
    "reshape_3": {},
    "permute_1": {},
    "multiply": {},
    "matmul_1": {},
    "reshape_9": {},
    "reshape_11": {},
    "matmul": {},
    "reshape_10": {},
    "reshape_4": {},
    "reshape_5": {},
    "reshape_6": {},
    "stack_concat": {},
    "softmax": {},
    "add": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_multi_attention)

r_ffn_2_fc = {
"ruler_name": "ffn_convs",
"src_ops_alias": ["Squeeze", "BiasAdd", "Reshape", "C", "MatMul", "C_1", "Reshape_1", "C_2", "Transpose", "C_3",
    "Relu", "C_4", "BiasAdd_1", "Reshape_2", "C_5", "MatMul_1", "C_6", "Reshape_3", "C_7", "Transpose_1", "C_8",
    "ExpandDims", "C_9", "C_10"],
"src_inter_flow": [["BiasAdd:out0", "Squeeze:in0"], ["Reshape:out0", "BiasAdd:in0"], ["C:out0", "BiasAdd:in1"],
    ["MatMul:out0", "Reshape:in0"], ["C_1:out0", "Reshape:in1"], ["Reshape_1:out0", "MatMul:in0"],
    ["C_2:out0", "MatMul:in1"], ["Transpose:out0", "Reshape_1:in0"], ["C_3:out0", "Reshape_1:in1"],
    ["Relu:out0", "Transpose:in0"], ["C_4:out0", "Transpose:in1"], ["BiasAdd_1:out0", "Relu:in0"],
    ["Reshape_2:out0", "BiasAdd_1:in0"], ["C_5:out0", "BiasAdd_1:in1"], ["MatMul_1:out0", "Reshape_2:in0"],
    ["C_6:out0", "Reshape_2:in1"], ["Reshape_3:out0", "MatMul_1:in0"], ["C_7:out0", "MatMul_1:in1"],
    ["Transpose_1:out0", "Reshape_3:in0"], ["C_8:out0", "Reshape_3:in1"], ["ExpandDims:out0", "Transpose_1:in0"],
    ["C_9:out0", "Transpose_1:in1"], ["C_10:out0", "ExpandDims:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["fullconnect", "relu", "fullconnect_1"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "fullconnect_1:out0"]],
"acu_inter_flow": [["fullconnect:out0", "relu:in0"], ["relu:out0", "fullconnect_1:in0"]],
"param_map": {
    "relu": {},
    "fullconnect": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_7:out0'])[1]"],
        'bias': ['BOOL', 'VALUE', True]},
    "fullconnect_1": {'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_2:out0'])[1]"],
        'bias': ['BOOL', 'VALUE', True]},
},
"blob_map": {
    "relu": {},
    "fullconnect": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_7:out0'])"],
                    'bias': ['CODE', "self.tensor_to_numpy(tensor['C_5:out0'])"],},
    "fullconnect_1": {'weight': ['CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                      'bias': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"], },
},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['C:out0'])) < 2 and " \
                 "len(self.shape_pick(tensor['C_5:out0'])) < 2"}
ruler_list.append(r_ffn_2_fc)

#rule for time major lstm without projection and peephole
r_time_major_lstm_1_10 = {
"ruler_name": "lstm_tf_1.10",
"src_ops_alias": ["TensorArrayGatherV3", "TensorArrayV3", "Range", "Exit", "C", "C_1", "TensorArraySizeV3",
    "C_2", "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "LogicalAnd", "TensorArrayWriteV3", "Less",
    "Less_1", "Enter_1", "Identity", "Mul", "Identity_1", "Merge_1", "Enter_2", "Merge_2", "Enter_3", "Switch_1",
    "Sigmoid", "Tanh", "Enter_4", "NextIteration_1", "Enter_5", "NextIteration_2", "C_3", "Split", "Add", "C_4",
    "Add_1", "C_5", "Add_2", "C_6", "BiasAdd", "Mul_1", "Mul_2", "Identity_2", "C_7", "C_8", "MatMul", "Enter_6",
    "Sigmoid_1", "Identity_3", "Sigmoid_2", "Tanh_1", "Switch_2", "ConcatV2", "Enter_7", "C_9", "Add_3", "Switch_3",
    "TensorArrayReadV3", "Identity_4", "C_10", "C_11", "C_12", "Merge_3", "Enter_8", "Enter_9", "Switch_4",
    "Enter_10", "NextIteration_3", "TensorArrayV3_1", "TensorArrayScatterV3", "Merge_4", "C_13", "C_14",
    "Enter_11", "NextIteration_4", "C_15"],
"src_inter_flow": [["Identity_2:out4096", "C_12:in0"], ["Add:out0", "Tanh:in0"], ["Tanh_1:out0", "Mul_2:in1"],
    ["Tanh:out0", "Mul:in1"], ["Merge:out0", "Switch:in0"], ["Enter_8:out0", "TensorArrayReadV3:in0"],
    ["Identity_2:out0", "Add_1:in0"], ["Enter_11:out0", "Merge_4:in0"], ["Sigmoid:out0", "Mul:in0"],
    ["Identity_2:out4096", "C_6:in0"], ["Mul_2:out0", "Add:in1"], ["Enter_10:out0", "Merge_3:in0"],
    ["C_15:out0", "Enter_11:in0"], ["C_9:out0", "Enter_6:in0"], ["Merge_1:out0", "Less:in0"],
    ["Switch_4:out1", "Identity_4:in0"], ["Add_2:out0", "NextIteration_2:in0"], ["C_11:out0", "Enter_7:in0"],
    ["Exit:out0", "TensorArraySizeV3:in1"], ["Sigmoid_2:out0", "Mul_2:in0"], ["Sigmoid_1:out0", "Mul_1:in0"],
    ["Identity_4:out0", "ConcatV2:in1"], ["Add_1:out0", "NextIteration_1:in0"], ["LoopCond:out0", "Switch_4:in1"],
    ["LoopCond:out0", "Switch_2:in1"], ["C:out0", "TensorArrayV3_1:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["Identity_3:out0", "Mul_1:in1"], ["Merge_4:out0", "Switch_4:in0"], ["Enter_2:out0", "Less:in1"],
    ["Less_1:out0", "LogicalAnd:in1"], ["Range:out0", "TensorArrayGatherV3:in1"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["TensorArrayScatterV3:out0", "Enter_9:in0"],
    ["Exit:out0", "TensorArrayGatherV3:in2"], ["C_2:out0", "Range:in2"], ["Mul:out0", "NextIteration_4:in0"],
    ["C_1:out0", "Range:in0"], ["Enter_3:out0", "Less_1:in1"], ["Merge_1:out0", "Switch_2:in0"],
    ["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["BiasAdd:out0", "Split:in1"],
    ["Enter_9:out0", "TensorArrayReadV3:in2"], ["Switch_1:out1", "Identity:in0"],
    ["TensorArraySizeV3:out0", "Range:in1"], ["LoopCond:out0", "Switch:in1"],
    ["LoopCond:out0", "Switch_3:in1"], ["Less:out0", "LogicalAnd:in0"], ["C:out0", "TensorArrayV3:in0"],
    ["Merge_2:out0", "Switch_1:in0"], ["ConcatV2:out0", "MatMul:in0"], ["Switch_3:out1", "Identity_3:in0"],
    ["Enter_4:out0", "Merge_1:in0"], ["TensorArrayV3:out1", "Enter:in0"],
    ["TensorArrayV3:out0", "TensorArraySizeV3:in0"], ["NextIteration_4:out0", "Merge_4:in1"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["Identity_2:out4096", "C_8:in0"],
    ["Split:out0", "Sigmoid_2:in0"], ["Enter_7:out0", "MatMul:in1"], ["Add_3:out0", "Sigmoid_1:in0"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"], ["Merge_3:out0", "Switch_3:in0"],
    ["Split:out1", "Tanh_1:in0"], ["Enter:out0", "Merge:in0"], ["C_6:out0", "Split:in0"],
    ["Add:out0", "NextIteration_3:in0"], ["Switch:out1", "Identity_1:in0"], ["C_3:out0", "Enter_3:in0"],
    ["Identity:out0", "Add_2:in0"], ["LogicalAnd:out0", "LoopCond:in0"], ["Enter_6:out0", "BiasAdd:in1"],
    ["MatMul:out0", "BiasAdd:in0"], ["C_13:out0", "Enter_10:in0"], ["C_10:out0", "ConcatV2:in2"],
    ["C_4:out0", "Enter_4:in0"], ["TensorArrayReadV3:out0", "ConcatV2:in0"], ["Identity_2:out4096", "C_7:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Identity:out0", "TensorArrayReadV3:in1"],
    ["C_7:out0", "Add_1:in1"], ["NextIteration:out0", "Merge:in1"], ["Split:out2", "Add_3:in0"],
    ["C_14:out0", "TensorArrayScatterV3:in1"], ["LoopCond:out0", "Switch_1:in1"], ["C_12:out0", "Add_3:in1"],
    ["Identity_2:out4096", "C_10:in0"], ["C_8:out0", "Add_2:in1"], ["C:out0", "Enter_2:in0"],
    ["Mul_1:out0", "Add:in0"], ["Enter_1:out0", "TensorArrayWriteV3:in0"], ["Switch:out0", "Exit:in0"],
    ["Mul:out0", "TensorArrayWriteV3:in2"], ["Enter_5:out0", "Merge_2:in0"],
    ["TensorArrayWriteV3:out0", "NextIteration:in0"], ["Switch_2:out1", "Identity_2:in0"],
    ["NextIteration_1:out0", "Merge_1:in1"], ["Identity_1:out0", "TensorArrayWriteV3:in3"],
    ["TensorArrayV3_1:out0", "Enter_8:in0"], ["C_5:out0", "Enter_5:in0"], ["Split:out3", "Sigmoid:in0"],
    ["TensorArrayV3:out0", "Enter_1:in0"], ["Merge_2:out0", "Less_1:in0"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["TensorArrayGatherV3:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "lstm:out0"]],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_12:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['Enter_7:out0'])[1] / 4"],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_11:out0'])"],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_9:out0'])"],
    }},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_time_major_lstm_1_10)

@rule_pyfunc_def
def r_lstm_get_seq(self, node, tensor, tensors_name):
    in_shape = self.shape_pick(tensor[tensors_name[0]])
    out_shape = self.shape_pick(tensor[tensors_name[1]])
    if len(in_shape) == len(out_shape):
        return True
    else:
        return False

r_time_major_lstm_false_seq = {
"ruler_name": "time_major_lstm_false_seq",
"src_ops_alias": ["Exit", "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "Less", "C", "Mul", "Merge_1",
    "Enter_1", "Tanh", "Sigmoid", "Enter_2", "NextIteration_1", "C_1", "Add", "Split", "C_2", "Add_1",
    "Mul_1", "Mul_2", "C_3", "BiasAdd", "Identity", "C_4", "Identity_1", "Sigmoid_1", "Sigmoid_2",
    "Tanh_1", "MatMul", "Enter_3", "Switch_1", "Switch_2", "Add_2", "ConcatV2", "Enter_4", "C_5",
    "Merge_2", "C_6", "TensorArrayReadV3", "Identity_2", "C_7", "C_8", "Enter_5", "NextIteration_2",
    "Enter_6", "Enter_7", "C_9", "TensorArrayV3", "TensorArrayScatterV3", "C_10"],
"src_inter_flow": [["Switch:out0", "Exit:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"], ["Less:out0", "LoopCond:in0"],
    ["C:out0", "Enter:in0"], ["Mul:out0", "NextIteration:in0"], ["Merge_1:out0", "Less:in0"],
    ["Enter_1:out0", "Less:in1"], ["Tanh:out0", "Mul:in0"], ["Sigmoid:out0", "Mul:in1"],
    ["Enter_2:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["C_1:out0", "Enter_1:in0"], ["Add:out0", "Tanh:in0"], ["Split:out3", "Sigmoid:in0"],
    ["C_2:out0", "Enter_2:in0"], ["Add_1:out0", "NextIteration_1:in0"], ["Mul_1:out0", "Add:in0"],
    ["Mul_2:out0", "Add:in1"], ["C_3:out0", "Split:in0"], ["BiasAdd:out0", "Split:in1"],
    ["Identity:out0", "Add_1:in0"], ["C_4:out0", "Add_1:in1"], ["Identity_1:out0", "Mul_1:in0"],
    ["Sigmoid_1:out0", "Mul_1:in1"], ["Sigmoid_2:out0", "Mul_2:in0"], ["Tanh_1:out0", "Mul_2:in1"],
    ["Identity:out4096", "C_3:in0"], ["MatMul:out0", "BiasAdd:in0"], ["Enter_3:out0", "BiasAdd:in1"],
    ["Switch_1:out1", "Identity:in0"], ["Identity:out4096", "C_4:in0"],
    ["Switch_2:out1", "Identity_1:in0"], ["Split:out0", "Sigmoid_2:in0"],
    ["Add_2:out0", "Sigmoid_1:in0"], ["Split:out1", "Tanh_1:in0"], ["LoopCond:out0", "Switch_1:in1"],
    ["Merge_1:out0", "Switch_1:in0"], ["LoopCond:out0", "Switch_2:in1"],
    ["ConcatV2:out0", "MatMul:in0"], ["Enter_4:out0", "MatMul:in1"], ["C_5:out0", "Enter_3:in0"],
    ["Split:out2", "Add_2:in0"], ["Merge_2:out0", "Switch_2:in0"], ["C_6:out0", "Add_2:in1"],
    ["TensorArrayReadV3:out0", "ConcatV2:in0"], ["Identity_2:out0", "ConcatV2:in1"],
    ["C_7:out0", "ConcatV2:in2"], ["C_8:out0", "Enter_4:in0"], ["Switch:out1", "Identity_2:in0"],
    ["Identity:out4096", "C_6:in0"], ["Identity:out0", "TensorArrayReadV3:in1"],
    ["Enter_5:out0", "Merge_2:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["Identity:out4096", "C_7:in0"], ["Enter_6:out0", "TensorArrayReadV3:in0"],
    ["Enter_7:out0", "TensorArrayReadV3:in2"], ["Add:out0", "NextIteration_2:in0"],
    ["C_9:out0", "Enter_5:in0"], ["C_1:out0", "TensorArrayV3:in0"],
    ["TensorArrayV3:out0", "Enter_6:in0"], ["TensorArrayScatterV3:out0", "Enter_7:in0"],
    ["TensorArrayV3:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3:out1", "TensorArrayScatterV3:in3"], ["C_10:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["Exit:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["Exit:out0", "lstm:out0"]],
"acu_inter_flow": [],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_6:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['Enter_4:out0'])[1] / 4"],
    'return_sequences': ['BOOL', 'PYFUNC', r_lstm_get_seq(tensors_name=['I:out0', 'Exit:out0'])],}},
"blob_map": {"lstm": {
    'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_8:out0'])"],
    'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_5:out0'])"],}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_time_major_lstm_false_seq)

r_time_major_lstm_false_seq_1_13 = {
"ruler_name": "lstm_time_major_no_seq_1_13",
"src_ops_alias": ["Exit", "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "LogicalAnd", "C", "Mul", "Less",
    "Less_1", "Sigmoid", "Tanh", "Merge_1", "Enter_1", "Merge_2", "Enter_2", "Split", "Add", "Enter_3",
    "NextIteration_1", "C_1", "Enter_4", "NextIteration_2", "C_2", "C_3", "BiasAdd", "Mul_1", "Mul_2",
    "C_4", "Add_1", "C_5", "Add_2", "Identity", "MatMul", "Enter_5", "Sigmoid_1", "Identity_1",
    "Sigmoid_2", "Tanh_1", "C_6", "Identity_2", "C_7", "Switch_1", "ConcatV2", "Enter_6", "C_8",
    "Add_3", "Switch_2", "Switch_3", "TensorArrayReadV3", "Identity_3", "C_9", "C_10", "C_11",
    "Merge_3", "Enter_7", "Enter_8", "Enter_9", "NextIteration_3", "TensorArrayV3",
    "TensorArrayScatterV3", "C_12", "C_13"],
"src_inter_flow": [["Switch:out0", "Exit:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"],
    ["LogicalAnd:out0", "LoopCond:in0"], ["C:out0", "Enter:in0"], ["Mul:out0", "NextIteration:in0"],
    ["Less:out0", "LogicalAnd:in0"], ["Less_1:out0", "LogicalAnd:in1"], ["Sigmoid:out0", "Mul:in0"],
    ["Tanh:out0", "Mul:in1"], ["Merge_1:out0", "Less:in0"], ["Enter_1:out0", "Less:in1"],
    ["Merge_2:out0", "Less_1:in0"], ["Enter_2:out0", "Less_1:in1"], ["Split:out3", "Sigmoid:in0"],
    ["Add:out0", "Tanh:in0"], ["Enter_3:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["C_1:out0", "Enter_1:in0"], ["Enter_4:out0", "Merge_2:in0"],
    ["NextIteration_2:out0", "Merge_2:in1"], ["C_2:out0", "Enter_2:in0"], ["C_3:out0", "Split:in0"],
    ["BiasAdd:out0", "Split:in1"], ["Mul_1:out0", "Add:in0"], ["Mul_2:out0", "Add:in1"],
    ["C_4:out0", "Enter_3:in0"], ["Add_1:out0", "NextIteration_1:in0"], ["C_5:out0", "Enter_4:in0"],
    ["Add_2:out0", "NextIteration_2:in0"], ["Identity:out4096", "C_3:in0"],
    ["MatMul:out0", "BiasAdd:in0"], ["Enter_5:out0", "BiasAdd:in1"], ["Sigmoid_1:out0", "Mul_1:in0"],
    ["Identity_1:out0", "Mul_1:in1"], ["Sigmoid_2:out0", "Mul_2:in0"], ["Tanh_1:out0", "Mul_2:in1"],
    ["Identity:out0", "Add_1:in0"], ["C_6:out0", "Add_1:in1"], ["Identity_2:out0", "Add_2:in0"],
    ["C_7:out0", "Add_2:in1"], ["Switch_1:out1", "Identity:in0"], ["ConcatV2:out0", "MatMul:in0"],
    ["Enter_6:out0", "MatMul:in1"], ["C_8:out0", "Enter_5:in0"], ["Split:out0", "Sigmoid_2:in0"],
    ["Add_3:out0", "Sigmoid_1:in0"], ["Split:out1", "Tanh_1:in0"], ["Switch_2:out1", "Identity_1:in0"],
    ["LoopCond:out0", "Switch_1:in1"], ["Identity:out4096", "C_6:in0"],
    ["Merge_1:out0", "Switch_1:in0"], ["Identity:out4096", "C_7:in0"],
    ["Switch_3:out1", "Identity_2:in0"], ["LoopCond:out0", "Switch_2:in1"],
    ["Split:out2", "Add_3:in0"], ["TensorArrayReadV3:out0", "ConcatV2:in0"],
    ["Identity_3:out0", "ConcatV2:in1"], ["C_9:out0", "ConcatV2:in2"],
    ["LoopCond:out0", "Switch_3:in1"], ["C_10:out0", "Enter_6:in0"], ["Merge_2:out0", "Switch_3:in0"],
    ["Switch:out1", "Identity_3:in0"], ["C_11:out0", "Add_3:in1"], ["Merge_3:out0", "Switch_2:in0"],
    ["Identity_2:out0", "TensorArrayReadV3:in1"], ["Identity:out4096", "C_9:in0"],
    ["Enter_7:out0", "TensorArrayReadV3:in0"], ["Enter_8:out0", "TensorArrayReadV3:in2"],
    ["Identity:out4096", "C_11:in0"], ["Add:out0", "NextIteration_3:in0"],
    ["Enter_9:out0", "Merge_3:in0"], ["NextIteration_3:out0", "Merge_3:in1"],
    ["TensorArrayV3:out0", "Enter_7:in0"], ["C_1:out0", "TensorArrayV3:in0"],
    ["TensorArrayScatterV3:out0", "Enter_8:in0"], ["C_12:out0", "Enter_9:in0"],
    ["TensorArrayV3:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3:out1", "TensorArrayScatterV3:in3"], ["C_13:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["Exit:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["Exit:out0", "lstm:out0"]],
"acu_inter_flow": [],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_11:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['Enter_6:out0'])[1] / 4"],
    'return_sequences': ['BOOL', 'PYFUNC', r_lstm_get_seq(tensors_name=['I:out0', 'Exit:out0'])],
}},
"blob_map": {"lstm": {
    'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_10:out0'])"],
    'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_8:out0'])"],
}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_time_major_lstm_false_seq_1_13)

@rule_pyfunc_def
def r_gru_get_bias(self, node, tensor, tensor_name):
    org_data = self.tensor_to_numpy(tensor[tensor_name])
    import numpy as np
    zeros_data = np.zeros_like(org_data)
    return np.concatenate((org_data, zeros_data), axis=0)

r_gru_time_major_encoder={
"ruler_name": "gru_time_major_encoder",
"src_ops_alias": ["TensorArrayGatherV3", "TensorArrayV3", "Range", "Exit", "C", "C_1", "TensorArraySizeV3", "C_2",
    "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "LogicalAnd", "TensorArrayWriteV3",
    "Less", "Less_1", "Enter_1", "Identity", "Select", "Identity_1", "Merge_1", "Enter_2", "Merge_2",
    "Enter_3", "Switch_1", "GreaterEqual", "Enter_4", "Add", "Enter_5", "NextIteration_1", "Enter_6",
    "NextIteration_2", "Minimum", "Enter_7", "C_3", "Mul", "Mul_1", "C_4", "Add_1", "C_5", "Add_2",
    "Maximum", "Identity_2", "Split", "Identity_3", "Sub", "Tanh", "Identity_4", "C_6", "C_7", "C_8",
    "Max", "C_9", "Sigmoid", "Switch_2", "C_10", "BiasAdd", "Switch_3", "C_11", "BiasAdd_1", "Merge_3",
    "MatMul", "Enter_8", "MatMul_1", "Enter_9", "Enter_10", "NextIteration_3", "ConcatV2", "Enter_11",
    "C_12", "ConcatV2_1", "Enter_12", "C_13", "C_14", "Select_1", "TensorArrayReadV3", "Mul_2", "C_15",
    "C_16", "C_17", "C_18", "Enter_13", "Enter_14", "TensorArrayV3_1", "TensorArrayScatterV3", "C_19"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["Range:out0", "TensorArrayGatherV3:in1"],
    ["Exit:out0", "TensorArrayGatherV3:in2"], ["C:out0", "TensorArrayV3:in0"],
    ["C_1:out0", "Range:in0"], ["TensorArraySizeV3:out0", "Range:in1"], ["C_2:out0", "Range:in2"],
    ["Switch:out0", "Exit:in0"], ["TensorArrayV3:out0", "TensorArraySizeV3:in0"],
    ["Exit:out0", "TensorArraySizeV3:in1"], ["Merge:out0", "Switch:in0"],
    ["LoopCond:out0", "Switch:in1"], ["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"],
    ["TensorArrayV3:out1", "Enter:in0"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["TensorArrayWriteV3:out0", "NextIteration:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["Less_1:out0", "LogicalAnd:in1"], ["Enter_1:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Select:out0", "TensorArrayWriteV3:in2"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["TensorArrayV3:out0", "Enter_1:in0"],
    ["Merge_1:out0", "Less:in0"], ["Enter_2:out0", "Less:in1"], ["Merge_2:out0", "Less_1:in0"],
    ["Enter_3:out0", "Less_1:in1"], ["Switch_1:out1", "Identity:in0"],
    ["Switch:out1", "Identity_1:in0"], ["GreaterEqual:out0", "Select:in0"],
    ["Enter_4:out0", "Select:in1"], ["Add:out0", "Select:in2"], ["C:out0", "Enter_2:in0"],
    ["Enter_5:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Enter_6:out0", "Merge_2:in0"],
    ["NextIteration_2:out0", "Merge_2:in1"], ["Minimum:out0", "Enter_3:in0"],
    ["Merge_2:out0", "Switch_1:in0"], ["Identity:out0", "GreaterEqual:in0"],
    ["Enter_7:out0", "GreaterEqual:in1"], ["C_3:out0", "Enter_4:in0"], ["Mul:out0", "Add:in0"],
    ["Mul_1:out0", "Add:in1"], ["C_4:out0", "Enter_5:in0"], ["C:out0", "Minimum:in0"],
    ["Add_1:out0", "NextIteration_1:in0"], ["C_5:out0", "Enter_6:in0"],
    ["Add_2:out0", "NextIteration_2:in0"], ["Maximum:out0", "Minimum:in1"],
    ["Identity_2:out0", "Enter_7:in0"], ["Split:out1", "Mul:in0"], ["Identity_3:out0", "Mul:in1"],
    ["Sub:out0", "Mul_1:in0"], ["Tanh:out0", "Mul_1:in1"], ["Identity:out0", "Add_2:in0"],
    ["Identity_4:out0", "Add_1:in0"], ["C_6:out0", "Add_1:in1"], ["C_7:out0", "Add_2:in1"],
    ["C_8:out0", "Maximum:in0"], ["Max:out0", "Maximum:in1"], ["C_9:out0", "Split:in0"],
    ["Sigmoid:out0", "Split:in1"], ["Split:out1", "Sub:in1"], ["Switch_2:out1", "Identity_3:in0"],
    ["C_10:out0", "Sub:in0"], ["BiasAdd:out0", "Tanh:in0"], ["Switch_3:out1", "Identity_4:in0"],
    ["Identity_4:out4096", "C_6:in0"], ["Identity_4:out4096", "C_7:in0"],
    ["LoopCond:out0", "Switch_2:in1"], ["Identity_2:out0", "Max:in0"],
    ["Identity_4:out4096", "C_9:in0"], ["C_11:out0", "Max:in1"], ["LoopCond:out0", "Switch_3:in1"],
    ["BiasAdd_1:out0", "Sigmoid:in0"], ["Merge_1:out0", "Switch_3:in0"],
    ["Identity_4:out4096", "C_10:in0"], ["Merge_3:out0", "Switch_2:in0"],
    ["MatMul:out0", "BiasAdd:in0"], ["Enter_8:out0", "BiasAdd:in1"],
    ["MatMul_1:out0", "BiasAdd_1:in0"], ["Enter_9:out0", "BiasAdd_1:in1"],
    ["Enter_10:out0", "Merge_3:in0"], ["NextIteration_3:out0", "Merge_3:in1"],
    ["ConcatV2:out0", "MatMul:in0"], ["Enter_11:out0", "MatMul:in1"], ["C_12:out0", "Enter_8:in0"],
    ["ConcatV2_1:out0", "MatMul_1:in0"], ["Enter_12:out0", "MatMul_1:in1"],
    ["C_13:out0", "Enter_9:in0"], ["C_14:out0", "Enter_10:in0"],
    ["Select_1:out0", "NextIteration_3:in0"], ["Identity_3:out0", "ConcatV2_1:in1"],
    ["TensorArrayReadV3:out0", "ConcatV2:in0"], ["Mul_2:out0", "ConcatV2:in1"],
    ["C_15:out0", "ConcatV2:in2"], ["C_16:out0", "Enter_11:in0"],
    ["GreaterEqual:out0", "Select_1:in0"], ["Add:out0", "Select_1:in2"],
    ["Identity:out0", "TensorArrayReadV3:in1"], ["TensorArrayReadV3:out0", "ConcatV2_1:in0"],
    ["C_17:out0", "ConcatV2_1:in2"], ["Identity_3:out0", "Select_1:in1"],
    ["C_18:out0", "Enter_12:in0"], ["Split:out0", "Mul_2:in0"], ["Identity_3:out0", "Mul_2:in1"],
    ["Identity_4:out4096", "C_15:in0"], ["Enter_13:out0", "TensorArrayReadV3:in0"],
    ["Enter_14:out0", "TensorArrayReadV3:in2"], ["C:out0", "TensorArrayV3_1:in0"],
    ["Identity_4:out4096", "C_17:in0"], ["TensorArrayV3_1:out0", "Enter_13:in0"],
    ["TensorArrayScatterV3:out0", "Enter_14:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"], ["C_19:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"], ["I_1:out0", "Identity_2:in0"]],
"src_out_tensor": ["TensorArrayGatherV3:out0"],
"acu_lys_alias": ["gru"],
"src_acu_in_tensor_map": [["I:out0", "gru:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "gru:out0"]],
"acu_inter_flow": [],
"param_map": {"gru": {
    'time_major': ['BOOL', 'VALUE', True],
    'num_units': ['INT', 'CODE', "self.shape_pick(tensor['Enter_12:out0'])[1] / 2"],
}},
"blob_map": {"gru": {
    'rnn/gru_cell/gates/kernel':['CODE', "self.tensor_to_numpy(tensor['C_18:out0'])"],
    'rnn/gru_cell/gates/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_13:out0')],
    'rnn/gru_cell/candidate/kernel':['CODE', "self.tensor_to_numpy(tensor['C_16:out0'])"],
    'rnn/gru_cell/candidate/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_12:out0')],
}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gru_time_major_encoder)

r_gru_time_major_post={
"ruler_name": "gru_time_major_post",
"src_ops_alias": ["TensorArrayGatherV3", "TensorArrayV3", "Range", "Exit", "C", "C_1", "TensorArraySizeV3", "C_2",
    "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "LogicalAnd", "TensorArrayWriteV3",
    "Less", "Less_1", "Enter_1", "Identity", "Add", "Identity_1", "Merge_1", "Enter_2", "Merge_2",
    "Enter_3", "Switch_1", "Mul", "Mul_1", "Enter_4", "NextIteration_1", "Enter_5", "NextIteration_2",
    "C_3", "Split", "Identity_2", "Sub", "Tanh", "C_4", "Add_1", "C_5", "Add_2", "C_6", "Sigmoid",
    "Switch_2", "C_7", "BiasAdd", "Identity_3", "C_8", "C_9", "BiasAdd_1", "Merge_3", "MatMul",
    "Enter_6", "Switch_3", "MatMul_1", "Enter_7", "Enter_8", "NextIteration_3", "ConcatV2", "Enter_9",
    "C_10", "ConcatV2_1", "Enter_10", "C_11", "C_12", "TensorArrayReadV3", "Mul_2", "C_13", "C_14",
    "C_15", "C_16", "Enter_11", "Enter_12", "TensorArrayV3_1", "TensorArrayScatterV3", "C_17"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["Range:out0", "TensorArrayGatherV3:in1"],
    ["Exit:out0", "TensorArrayGatherV3:in2"], ["C:out0", "TensorArrayV3:in0"],
    ["C_1:out0", "Range:in0"], ["TensorArraySizeV3:out0", "Range:in1"], ["C_2:out0", "Range:in2"],
    ["Switch:out0", "Exit:in0"], ["TensorArrayV3:out0", "TensorArraySizeV3:in0"],
    ["Exit:out0", "TensorArraySizeV3:in1"], ["Merge:out0", "Switch:in0"],
    ["LoopCond:out0", "Switch:in1"], ["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"],
    ["TensorArrayV3:out1", "Enter:in0"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["TensorArrayWriteV3:out0", "NextIteration:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["Less_1:out0", "LogicalAnd:in1"], ["Enter_1:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Add:out0", "TensorArrayWriteV3:in2"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["TensorArrayV3:out0", "Enter_1:in0"],
    ["Merge_1:out0", "Less:in0"], ["Enter_2:out0", "Less:in1"], ["Merge_2:out0", "Less_1:in0"],
    ["Enter_3:out0", "Less_1:in1"], ["Switch_1:out1", "Identity:in0"],
    ["Switch:out1", "Identity_1:in0"], ["Mul:out0", "Add:in0"], ["Mul_1:out0", "Add:in1"],
    ["C:out0", "Enter_2:in0"], ["Enter_4:out0", "Merge_1:in0"],
    ["NextIteration_1:out0", "Merge_1:in1"], ["LoopCond:out0", "Switch_1:in1"],
    ["Enter_5:out0", "Merge_2:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["C_3:out0", "Enter_3:in0"], ["Merge_2:out0", "Switch_1:in0"], ["Split:out1", "Mul:in0"],
    ["Identity_2:out0", "Mul:in1"], ["Sub:out0", "Mul_1:in0"], ["Tanh:out0", "Mul_1:in1"],
    ["C_4:out0", "Enter_4:in0"], ["Add_1:out0", "NextIteration_1:in0"], ["C_5:out0", "Enter_5:in0"],
    ["Add_2:out0", "NextIteration_2:in0"], ["C_6:out0", "Split:in0"], ["Sigmoid:out0", "Split:in1"],
    ["Split:out1", "Sub:in1"], ["Switch_2:out1", "Identity_2:in0"], ["C_7:out0", "Sub:in0"],
    ["BiasAdd:out0", "Tanh:in0"], ["Identity:out0", "Add_2:in0"], ["Identity_3:out0", "Add_1:in0"],
    ["C_8:out0", "Add_1:in1"], ["LoopCond:out0", "Switch_2:in1"], ["C_9:out0", "Add_2:in1"],
    ["Identity_3:out4096", "C_6:in0"], ["BiasAdd_1:out0", "Sigmoid:in0"],
    ["Merge_3:out0", "Switch_2:in0"], ["Identity_3:out4096", "C_7:in0"],
    ["MatMul:out0", "BiasAdd:in0"], ["Enter_6:out0", "BiasAdd:in1"],
    ["Switch_3:out1", "Identity_3:in0"], ["Identity_3:out4096", "C_8:in0"],
    ["Identity_3:out4096", "C_9:in0"], ["LoopCond:out0", "Switch_3:in1"],
    ["MatMul_1:out0", "BiasAdd_1:in0"], ["Enter_7:out0", "BiasAdd_1:in1"],
    ["Merge_1:out0", "Switch_3:in0"], ["Enter_8:out0", "Merge_3:in0"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["ConcatV2:out0", "MatMul:in0"],
    ["Enter_9:out0", "MatMul:in1"], ["C_10:out0", "Enter_6:in0"], ["Add:out0", "NextIteration_3:in0"],
    ["ConcatV2_1:out0", "MatMul_1:in0"], ["Enter_10:out0", "MatMul_1:in1"],
    ["C_11:out0", "Enter_7:in0"], ["C_12:out0", "Enter_8:in0"], ["Identity_2:out0", "ConcatV2_1:in1"],
    ["TensorArrayReadV3:out0", "ConcatV2:in0"], ["Mul_2:out0", "ConcatV2:in1"],
    ["C_13:out0", "ConcatV2:in2"], ["C_14:out0", "Enter_9:in0"],
    ["Identity:out0", "TensorArrayReadV3:in1"], ["TensorArrayReadV3:out0", "ConcatV2_1:in0"],
    ["C_15:out0", "ConcatV2_1:in2"], ["C_16:out0", "Enter_10:in0"], ["Split:out0", "Mul_2:in0"],
    ["Identity_2:out0", "Mul_2:in1"], ["Identity_3:out4096", "C_13:in0"],
    ["Enter_11:out0", "TensorArrayReadV3:in0"], ["Enter_12:out0", "TensorArrayReadV3:in2"],
    ["C:out0", "TensorArrayV3_1:in0"], ["Identity_3:out4096", "C_15:in0"],
    ["TensorArrayV3_1:out0", "Enter_11:in0"], ["TensorArrayScatterV3:out0", "Enter_12:in0"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["C_17:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["TensorArrayGatherV3:out0"],
"acu_lys_alias": ["gru"],
"src_acu_in_tensor_map": [["I:out0", "gru:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "gru:out0"]],
"acu_inter_flow": [],
"param_map": {"gru": {
    'time_major': ['BOOL', 'VALUE', True],
    'num_units': ['INT', 'CODE', "self.shape_pick(tensor['Enter_10:out0'])[1] / 2"],
}},
"blob_map": {"gru": {
    'rnn/gru_cell/gates/kernel':['CODE', "self.tensor_to_numpy(tensor['C_16:out0'])"],
    'rnn/gru_cell/gates/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_11:out0')],
    'rnn/gru_cell/candidate/kernel':['CODE', "self.tensor_to_numpy(tensor['C_14:out0'])"],
    'rnn/gru_cell/candidate/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_10:out0')],
}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gru_time_major_post)

r_gru_out_hstate_1_13_2 = {
"ruler_name": "gru_out_hstate_1_13_2",
"src_ops_alias": ["TensorArrayGatherV3", "Exit", "TensorArrayV3", "Range", "Exit_1", "Switch", "C", "C_1",
    "TensorArraySizeV3", "C_2", "Switch_1", "Merge", "LoopCond", "Merge_1", "Enter", "NextIteration",
    "LogicalAnd", "Enter_1", "NextIteration_1", "C_3", "Add", "Less", "Less_1", "TensorArrayWriteV3",
    "Mul", "Mul_1", "Merge_2", "Enter_2", "Merge_3", "Enter_3", "Enter_4", "Identity", "Identity_1",
    "Split", "Identity_2", "Sub", "Tanh", "Enter_5", "NextIteration_2", "Enter_6", "NextIteration_3",
    "C_4", "Switch_2", "C_5", "Sigmoid", "C_6", "BiasAdd", "C_7", "Add_1", "C_8", "Add_2",
    "Identity_3", "BiasAdd_1", "MatMul", "Enter_7", "C_9", "C_10", "Switch_3", "MatMul_1", "Enter_8",
    "ConcatV2", "Enter_9", "C_11", "ConcatV2_1", "Enter_10", "C_12", "TensorArrayReadV3", "Mul_2",
    "C_13", "C_14", "C_15", "C_16", "Enter_11", "Enter_12", "TensorArrayV3_1", "TensorArrayScatterV3",
    "C_17"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["Range:out0", "TensorArrayGatherV3:in1"],
    ["Exit_1:out0", "TensorArrayGatherV3:in2"], ["Switch:out0", "Exit:in0"],
    ["C:out0", "TensorArrayV3:in0"], ["C_1:out0", "Range:in0"],
    ["TensorArraySizeV3:out0", "Range:in1"], ["C_2:out0", "Range:in2"],
    ["Switch_1:out0", "Exit_1:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["TensorArrayV3:out0", "TensorArraySizeV3:in0"], ["Exit_1:out0", "TensorArraySizeV3:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Merge_1:out0", "Switch_1:in0"], ["Enter:out0", "Merge:in0"],
    ["NextIteration:out0", "Merge:in1"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["Enter_1:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["C_3:out0", "Enter:in0"], ["Add:out0", "NextIteration:in0"],
    ["TensorArrayV3:out1", "Enter_1:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["Less_1:out0", "LogicalAnd:in1"], ["TensorArrayWriteV3:out0", "NextIteration_1:in0"],
    ["Mul:out0", "Add:in0"], ["Mul_1:out0", "Add:in1"], ["Merge_2:out0", "Less:in0"],
    ["Enter_2:out0", "Less:in1"], ["Merge_3:out0", "Less_1:in0"], ["Enter_3:out0", "Less_1:in1"],
    ["Add:out0", "TensorArrayWriteV3:in2"], ["Enter_4:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Identity_1:out0", "TensorArrayWriteV3:in3"],
    ["Split:out1", "Mul:in0"], ["Identity_2:out0", "Mul:in1"], ["C:out0", "Enter_2:in0"],
    ["Sub:out0", "Mul_1:in0"], ["Tanh:out0", "Mul_1:in1"], ["Enter_5:out0", "Merge_2:in0"],
    ["NextIteration_2:out0", "Merge_2:in1"], ["TensorArrayV3:out0", "Enter_4:in0"],
    ["Enter_6:out0", "Merge_3:in0"], ["NextIteration_3:out0", "Merge_3:in1"],
    ["Switch_1:out1", "Identity_1:in0"], ["C_4:out0", "Enter_3:in0"],
    ["Switch:out1", "Identity_2:in0"], ["Switch_2:out1", "Identity:in0"], ["C_5:out0", "Split:in0"],
    ["Sigmoid:out0", "Split:in1"], ["Split:out1", "Sub:in1"], ["C_6:out0", "Sub:in0"],
    ["BiasAdd:out0", "Tanh:in0"], ["C_7:out0", "Enter_5:in0"], ["Add_1:out0", "NextIteration_2:in0"],
    ["LoopCond:out0", "Switch_2:in1"], ["C_8:out0", "Enter_6:in0"], ["Merge_3:out0", "Switch_2:in0"],
    ["Add_2:out0", "NextIteration_3:in0"], ["Identity_3:out4096", "C_5:in0"],
    ["BiasAdd_1:out0", "Sigmoid:in0"], ["Identity_3:out4096", "C_6:in0"],
    ["MatMul:out0", "BiasAdd:in0"], ["Enter_7:out0", "BiasAdd:in1"], ["Identity_3:out0", "Add_1:in0"],
    ["Identity:out0", "Add_2:in0"], ["C_9:out0", "Add_1:in1"], ["C_10:out0", "Add_2:in1"],
    ["Switch_3:out1", "Identity_3:in0"], ["MatMul_1:out0", "BiasAdd_1:in0"],
    ["Enter_8:out0", "BiasAdd_1:in1"], ["LoopCond:out0", "Switch_3:in1"],
    ["ConcatV2:out0", "MatMul:in0"], ["Enter_9:out0", "MatMul:in1"], ["Merge_2:out0", "Switch_3:in0"],
    ["Identity_3:out4096", "C_9:in0"], ["C_11:out0", "Enter_7:in0"],
    ["Identity_3:out4096", "C_10:in0"], ["ConcatV2_1:out0", "MatMul_1:in0"],
    ["Enter_10:out0", "MatMul_1:in1"], ["C_12:out0", "Enter_8:in0"],
    ["Identity_2:out0", "ConcatV2_1:in1"], ["TensorArrayReadV3:out0", "ConcatV2:in0"],
    ["Mul_2:out0", "ConcatV2:in1"], ["C_13:out0", "ConcatV2:in2"], ["C_14:out0", "Enter_9:in0"],
    ["Identity:out0", "TensorArrayReadV3:in1"], ["TensorArrayReadV3:out0", "ConcatV2_1:in0"],
    ["C_15:out0", "ConcatV2_1:in2"], ["Split:out0", "Mul_2:in0"], ["Identity_2:out0", "Mul_2:in1"],
    ["C_16:out0", "Enter_10:in0"], ["Identity_3:out4096", "C_13:in0"],
    ["Enter_11:out0", "TensorArrayReadV3:in0"], ["Enter_12:out0", "TensorArrayReadV3:in2"],
    ["C:out0", "TensorArrayV3_1:in0"], ["Identity_3:out4096", "C_15:in0"],
    ["TensorArrayV3_1:out0", "Enter_11:in0"], ["TensorArrayScatterV3:out0", "Enter_12:in0"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["C_17:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["TensorArrayGatherV3:out0", "Exit:out0"],
"acu_lys_alias": ["gru"],
"src_acu_in_tensor_map": [["I:out0", "gru:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "gru:out0"], ["Exit:out0", "gru:out1"]],
"acu_inter_flow": [],
"param_map": {"gru": {'time_major': ['BOOL', 'VALUE', True],
    'num_units': ['INT', 'CODE', "self.shape_pick(tensor['Enter_10:out0'])[1] / 2"],}},
"blob_map": {"gru": {
    'rnn/gru_cell/gates/kernel':['CODE', "self.tensor_to_numpy(tensor['C_16:out0'])"],
    'rnn/gru_cell/gates/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_12:out0')],
    'rnn/gru_cell/candidate/kernel':['CODE', "self.tensor_to_numpy(tensor['C_14:out0'])"],
    'rnn/gru_cell/candidate/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_11:out0')],
}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gru_out_hstate_1_13_2)

r_gru_false_seq_1_13_2 = {
"ruler_name": "gru_false_seq_1_13_2",
"src_ops_alias": ["Exit", "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "LogicalAnd", "C", "Add", "Less",
    "Less_1", "Mul", "Mul_1", "Merge_1", "Enter_1", "Merge_2", "Enter_2", "Split", "Identity", "Sub",
    "Tanh", "Enter_3", "NextIteration_1", "C_1", "Enter_4", "NextIteration_2", "C_2", "C_3", "Sigmoid",
    "C_4", "BiasAdd", "C_5", "Add_1", "C_6", "Add_2", "Identity_1", "BiasAdd_1", "MatMul", "Enter_5",
    "C_7", "Identity_2", "C_8", "Switch_1", "MatMul_1", "Enter_6", "ConcatV2", "Enter_7", "C_9",
    "Switch_2", "ConcatV2_1", "Enter_8", "C_10", "TensorArrayReadV3", "Mul_2", "C_11", "C_12", "C_13",
    "C_14", "Enter_9", "Enter_10", "TensorArrayV3", "TensorArrayScatterV3", "C_15"],
"src_inter_flow": [["Switch:out0", "Exit:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"],
    ["LogicalAnd:out0", "LoopCond:in0"], ["C:out0", "Enter:in0"], ["Add:out0", "NextIteration:in0"],
    ["Less:out0", "LogicalAnd:in0"], ["Less_1:out0", "LogicalAnd:in1"], ["Mul:out0", "Add:in0"],
    ["Mul_1:out0", "Add:in1"], ["Merge_1:out0", "Less:in0"], ["Enter_1:out0", "Less:in1"],
    ["Merge_2:out0", "Less_1:in0"], ["Enter_2:out0", "Less_1:in1"], ["Split:out1", "Mul:in0"],
    ["Identity:out0", "Mul:in1"], ["Sub:out0", "Mul_1:in0"], ["Tanh:out0", "Mul_1:in1"],
    ["Enter_3:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["C_1:out0", "Enter_1:in0"], ["Enter_4:out0", "Merge_2:in0"],
    ["NextIteration_2:out0", "Merge_2:in1"], ["Switch:out1", "Identity:in0"],
    ["C_2:out0", "Enter_2:in0"], ["C_3:out0", "Split:in0"], ["Sigmoid:out0", "Split:in1"],
    ["Split:out1", "Sub:in1"], ["C_4:out0", "Sub:in0"], ["BiasAdd:out0", "Tanh:in0"],
    ["C_5:out0", "Enter_3:in0"], ["Add_1:out0", "NextIteration_1:in0"], ["C_6:out0", "Enter_4:in0"],
    ["Add_2:out0", "NextIteration_2:in0"], ["Identity_1:out4096", "C_3:in0"],
    ["BiasAdd_1:out0", "Sigmoid:in0"], ["Identity_1:out4096", "C_4:in0"],
    ["MatMul:out0", "BiasAdd:in0"], ["Enter_5:out0", "BiasAdd:in1"], ["Identity_1:out0", "Add_1:in0"],
    ["C_7:out0", "Add_1:in1"], ["Identity_2:out0", "Add_2:in0"], ["C_8:out0", "Add_2:in1"],
    ["Switch_1:out1", "Identity_1:in0"], ["MatMul_1:out0", "BiasAdd_1:in0"],
    ["Enter_6:out0", "BiasAdd_1:in1"], ["ConcatV2:out0", "MatMul:in0"], ["Enter_7:out0", "MatMul:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Identity_1:out4096", "C_7:in0"], ["C_9:out0", "Enter_5:in0"],
    ["Merge_1:out0", "Switch_1:in0"], ["Identity_1:out4096", "C_8:in0"],
    ["Switch_2:out1", "Identity_2:in0"], ["ConcatV2_1:out0", "MatMul_1:in0"],
    ["Enter_8:out0", "MatMul_1:in1"], ["LoopCond:out0", "Switch_2:in1"], ["C_10:out0", "Enter_6:in0"],
    ["Merge_2:out0", "Switch_2:in0"], ["TensorArrayReadV3:out0", "ConcatV2:in0"],
    ["Mul_2:out0", "ConcatV2:in1"], ["C_11:out0", "ConcatV2:in2"], ["Identity:out0", "ConcatV2_1:in1"],
    ["C_12:out0", "Enter_7:in0"], ["TensorArrayReadV3:out0", "ConcatV2_1:in0"],
    ["C_13:out0", "ConcatV2_1:in2"], ["Split:out0", "Mul_2:in0"], ["Identity:out0", "Mul_2:in1"],
    ["C_14:out0", "Enter_8:in0"], ["Identity_2:out0", "TensorArrayReadV3:in1"],
    ["Identity_1:out4096", "C_11:in0"], ["Enter_9:out0", "TensorArrayReadV3:in0"],
    ["Enter_10:out0", "TensorArrayReadV3:in2"], ["Identity_1:out4096", "C_13:in0"],
    ["C_1:out0", "TensorArrayV3:in0"], ["TensorArrayV3:out0", "Enter_9:in0"],
    ["TensorArrayScatterV3:out0", "Enter_10:in0"], ["TensorArrayV3:out1", "TensorArrayScatterV3:in3"],
    ["TensorArrayV3:out0", "TensorArrayScatterV3:in0"], ["C_15:out0", "TensorArrayScatterV3:in1"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["Exit:out0"],
"acu_lys_alias": ["gru"],
"src_acu_in_tensor_map": [["I:out0", "gru:in0"]],
"src_acu_out_tensor_map": [["Exit:out0", "gru:out0"]],
"acu_inter_flow": [],
"param_map": {"gru": {'time_major': ['BOOL', 'VALUE', True],
    'num_units': ['INT', 'CODE', "self.shape_pick(tensor['Enter_8:out0'])[1] / 2"],
    'return_sequences': ['BOOL', 'VALUE', False], }},
"blob_map": {"gru": {
    'rnn/gru_cell/gates/kernel':['CODE', "self.tensor_to_numpy(tensor['C_14:out0'])"],
    'rnn/gru_cell/gates/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_10:out0')],
    'rnn/gru_cell/candidate/kernel':['CODE', "self.tensor_to_numpy(tensor['C_12:out0'])"],
    'rnn/gru_cell/candidate/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_9:out0')],
}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gru_false_seq_1_13_2)

r_gru_cell={
"ruler_name": "gru_cell",
"src_ops_alias": ["Add", "Mul", "Mul_1", "Split", "Sub", "Tanh", "C", "Sigmoid", "C_1", "BiasAdd", "BiasAdd_1",
    "MatMul", "C_2", "MatMul_1", "C_3", "ConcatV2", "C_4", "ConcatV2_1", "C_5", "Mul_2", "C_6", "C_7"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Mul_1:out0", "Add:in1"], ["Split:out1", "Mul:in0"],
    ["Sub:out0", "Mul_1:in0"], ["Tanh:out0", "Mul_1:in1"], ["C:out0", "Split:in0"],
    ["Sigmoid:out0", "Split:in1"], ["Split:out1", "Sub:in1"], ["C_1:out0", "Sub:in0"],
    ["BiasAdd:out0", "Tanh:in0"], ["BiasAdd_1:out0", "Sigmoid:in0"], ["MatMul:out0", "BiasAdd:in0"],
    ["C_2:out0", "BiasAdd:in1"], ["MatMul_1:out0", "BiasAdd_1:in0"], ["C_3:out0", "BiasAdd_1:in1"],
    ["ConcatV2:out0", "MatMul:in0"], ["C_4:out0", "MatMul:in1"], ["ConcatV2_1:out0", "MatMul_1:in0"],
    ["C_5:out0", "MatMul_1:in1"], ["Mul_2:out0", "ConcatV2:in1"], ["C_6:out0", "ConcatV2:in2"],
    ["C_7:out0", "ConcatV2_1:in2"], ["Split:out0", "Mul_2:in0"]],
"src_in_anchor": [["I:out0", "ConcatV2_1:in0"], ["I:out0", "ConcatV2:in0"], ["I_1:out0", "ConcatV2_1:in1"],
    ["I_1:out0", "Mul_2:in1"], ["I_1:out0", "Mul:in1"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["gru_cell"],
"src_acu_in_tensor_map": [["I:out0", "gru_cell:in0"], ["I_1:out0", "gru_cell:in1"]],
"src_acu_out_tensor_map": [["Add:out0", "gru_cell:out0"], ["Add:out0", "gru_cell:out1"]],
"acu_inter_flow": [],
"param_map": {"gru_cell": {
    'num_units': ['INT', 'CODE', "self.shape_pick(tensor['C_5:out0'])[1] / 2"],
}},
"blob_map": {"gru_cell": {
    'gates/kernel':['CODE', "self.tensor_to_numpy(tensor['C_5:out0'])"],
    'gates/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_3:out0')],
    'candidate/kernel':['CODE', "self.tensor_to_numpy(tensor['C_4:out0'])"],
    'candidate/bias':['PYFUNC', r_gru_get_bias(tensor_name='C_2:out0')],
}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gru_cell)

@rule_pyfunc_def
def r_embedding_lookup_pre_condition(self, node, tensor, axis_tensor_name):
    axis = self.tensor_to_numpy(tensor[axis_tensor_name])
    return axis == 0

r_embedding_lookup={
"ruler_name": "embedding_lookup",
"src_ops_alias": ["GatherV2", "C", "C_1"],
"src_inter_flow": [["C:out0", "GatherV2:in0"], ["C_1:out0", "GatherV2:in2"]],
"src_in_anchor": [["I:out0", "GatherV2:in1"]],
"src_out_tensor": ["GatherV2:out0"],
"acu_lys_alias": ["embedding_lookup"],
"src_acu_in_tensor_map": [["I:out0", "embedding_lookup:in0"]],
"src_acu_out_tensor_map": [["GatherV2:out0", "embedding_lookup:out0"]],
"acu_inter_flow": [],
"param_map": {"embedding_lookup": {}},
"blob_map": {"embedding_lookup": {
    'embedding_params': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
}},
"priority_tip": 0,
"pre_condition": r_embedding_lookup_pre_condition(axis_tensor_name='C_1:out0')}
#ruler_list.append(r_embedding_lookup)

@rule_pyfunc_def
def r_reverse_sequence_pre_condition(self, node, tensor, input_name):
    input_tensor_shape = self.shape_pick(tensor[input_name])
    return len(input_tensor_shape) == 1

r_reverse_sequence_to_reverse={
"ruler_name": "reverse_sequence_to_reverse",
"src_ops_alias": ["ReverseSequence"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ReverseSequence:in0"], ["I_1:out0", "ReverseSequence:in1"]],
"src_out_tensor": ["ReverseSequence:out0"],
"acu_lys_alias": ["reverse"],
"src_acu_in_tensor_map": [["I:out0", "reverse:in0"]],
"src_acu_out_tensor_map": [["ReverseSequence:out0", "reverse:out0"]],
"acu_inter_flow": [],
"param_map": {"reverse": {
    'axis': ['INTS', 'VALUE', [1]],
}},
"blob_map": {"reverse": {}},
"priority_tip": 0,
"pre_condition": r_reverse_sequence_pre_condition(input_name='I_1:out0')}
ruler_list.append(r_reverse_sequence_to_reverse)

@rule_pyfunc_def
def r_relu20_get_n(self, node, tensor, tensor_name):
    return self.tensor_to_numpy(tensor[tensor_name])

r_relu20 = {
"ruler_name": "relu20",
"src_ops_alias": ["Mul", "RealDiv", "C", "Relu6", "C_1", "Mul_1", "RealDiv_1", "C_2", "C_3"],
"src_inter_flow": [["C:out0", "Mul:in1"], ["C_1:out0", "RealDiv:in1"], ["RealDiv_1:out0", "Mul_1:in0"],
    ["Mul_1:out0", "Relu6:in0"], ["C_3:out0", "RealDiv_1:in1"], ["RealDiv:out0", "Mul:in0"],
    ["Relu6:out0", "RealDiv:in0"], ["C_2:out0", "Mul_1:in1"]],
"src_in_anchor": [["I:out0", "RealDiv_1:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["relun"],
"src_acu_in_tensor_map": [["I:out0", "relun:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "relun:out0"]],
"param_map": {"relun": {
    'relu_clamp_top': ['INT', 'PYFUNC', r_relu20_get_n(tensor_name='C_3:out0')],
    }},
"blob_map": {"relun": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None }
ruler_list.append(r_relu20)
#Mul:mul_3;RealDiv:truediv_3;C:mul_3/y;Relu6:relu2;C_1:truediv_3/y;
#Mul_1:mul_2;RealDiv_1:truediv_2;
#C_2:mul_2/y;C_3:truediv_2/y

@rule_pyfunc_def
def r_lstmunit_rule_lstm_to_lstmunit_pre_condition(self, node, tensor, input_name):
    input_tensor_shape = self.shape_pick(tensor[input_name])
    return input_tensor_shape[1] == 1 # convert single timestep lstm to lstmunit

@rule_pyfunc_def
def r_lstmunit_rule_lstm_to_lstmunit_reshape1_shape(self, node, tensor, input_name):
    input_tensor_shape = self.shape_pick(tensor[input_name])
    batch = 0 if input_tensor_shape[0] == 1 else input_tensor_shape[0]
    return [batch, input_tensor_shape[2]]

@rule_pyfunc_def
def r_lstmunit_rule_lstm_to_lstmunit_reshape2_shape(self, node, tensor, input_name, kernel_name):
    input_tensor_shape = self.shape_pick(tensor[input_name])
    weights = self.shape_pick(tensor[kernel_name])[1] / 4
    batch = 0 if input_tensor_shape[0] == 1 else input_tensor_shape[0]
    return [batch, input_tensor_shape[1], int(weights)]

# convert single timestep lstm to lstmunit, add reshape to make shape match
r_lstmunit_rule_lstm_to_lstmunit = {
"ruler_name": "single_timestep_lstm_to_lstmunit",
"src_ops_alias": ["Transpose", "TensorArrayGatherV3", "C", "TensorArrayV3", "Range",
    "Exit", "C_1", "C_2", "TensorArraySizeV3", "C_3", "Switch", "Merge", "LoopCond",
    "Enter", "NextIteration", "LogicalAnd", "TensorArrayWriteV3", "Less", "Less_1",
    "Enter_1", "Identity", "Mul", "Identity_1", "Merge_1", "Enter_2", "Merge_2", "Enter_3",
    "Switch_1", "Sigmoid", "Tanh", "Enter_4", "NextIteration_1", "Enter_5", "NextIteration_2",
    "C_4", "Split", "Add", "C_5", "Add_1", "C_6", "Add_2", "C_7", "BiasAdd", "Mul_1", "Mul_2",
    "Identity_2", "C_8", "C_9", "MatMul", "Enter_6", "Sigmoid_1", "Identity_3", "Sigmoid_2", "Tanh_1",
    "Switch_2", "ConcatV2", "Enter_7", "C_10", "Add_3", "Switch_3", "TensorArrayReadV3", "Identity_4",
    "C_11", "C_12", "C_13", "Merge_3", "Enter_8", "Enter_9", "Switch_4", "Enter_10", "NextIteration_3",
    "TensorArrayV3_1", "TensorArrayScatterV3", "Merge_4", "C_14", "Transpose_1", "Enter_11", "NextIteration_4",
    "C_15", "Exit_1", "Exit_2"],
"src_inter_flow": [["C_1:out0", "TensorArrayV3:in0"], ["Exit:out0", "TensorArraySizeV3:in1"],
    ["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["LoopCond:out0", "Switch_2:in1"],
    ["Merge_3:out0", "Switch_3:in0"], ["Mul_2:out0", "Add:in1"], ["C_6:out0", "Enter_5:in0"],
    ["Split:out3", "Sigmoid:in0"], ["Identity_4:out0", "ConcatV2:in1"], ["Enter:out0", "Merge:in0"],
    ["Enter_5:out0", "Merge_2:in0"], ["C_14:out0", "TensorArrayScatterV3:in1"], ["C_9:out0", "Add_2:in1"],
    ["Add:out0", "Tanh:in0"], ["Identity:out0", "TensorArrayReadV3:in1"], ["Tanh:out0", "Mul:in1"],
    ["Sigmoid:out0", "Mul:in0"], ["ConcatV2:out0", "MatMul:in0"], ["Add:out0", "NextIteration_3:in0"],
    ["Add_3:out0", "Sigmoid_1:in0"], ["Identity:out0", "Add_2:in0"], ["NextIteration_4:out0", "Merge_4:in1"],
    ["Merge_2:out0", "Less_1:in0"], ["Sigmoid_1:out0", "Mul_1:in0"], ["Enter_11:out0", "Merge_4:in0"],
    ["Add_2:out0", "NextIteration_2:in0"], ["BiasAdd:out0", "Split:in1"], ["Enter_6:out0", "BiasAdd:in1"],
    ["Switch:out0", "Exit:in0"], ["Switch_3:out1", "Identity_3:in0"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["C_12:out0", "Enter_7:in0"], ["Enter_4:out0", "Merge_1:in0"], ["LoopCond:out0", "Switch:in1"],
    ["C_13:out0", "Add_3:in1"], ["TensorArrayWriteV3:out0", "NextIteration:in0"],
    ["TensorArraySizeV3:out0", "Range:in1"], ["Less_1:out0", "LogicalAnd:in1"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["LoopCond:out0", "Switch_4:in1"], ["Split:out2", "Add_3:in0"], ["Identity:out0", "TensorArrayWriteV3:in1"],
    ["Switch_1:out1", "Identity:in0"], ["NextIteration_2:out0", "Merge_2:in1"], ["C_10:out0", "Enter_6:in0"],
    ["NextIteration_1:out0", "Merge_1:in1"], ["Enter_7:out0", "MatMul:in1"],
    ["TensorArrayReadV3:out0", "ConcatV2:in0"],
    ["C_1:out0", "Enter_2:in0"], ["Switch:out1", "Identity_1:in0"], ["C_11:out0", "ConcatV2:in2"],
    ["Split:out0", "Sigmoid_2:in0"], ["C_8:out0", "Add_1:in1"], ["Identity_2:out0", "Add_1:in0"],
    ["Enter_3:out0", "Less_1:in1"], ["Identity_1:out0", "TensorArrayWriteV3:in3"],
    ["TensorArrayV3_1:out0", "Enter_8:in0"],
    ["C_4:out0", "Enter_3:in0"], ["C_3:out0", "Range:in2"], ["Exit:out0", "TensorArrayGatherV3:in2"],
    ["Merge_1:out0", "Less:in0"], ["Identity_2:out4096", "C_11:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["C_7:out0", "Split:in0"], ["Sigmoid_2:out0", "Mul_2:in0"], ["Tanh_1:out0", "Mul_2:in1"],
    ["C_5:out0", "Enter_4:in0"],
    ["C_15:out0", "Transpose_1:in1"], ["Switch_4:out0", "Exit_1:in0"], ["Mul_1:out0", "Add:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["Split:out1", "Tanh_1:in0"],
    ["Identity_2:out4096", "C_7:in0"],
    ["Merge_4:out0", "Switch_4:in0"], ["Switch_2:out1", "Identity_2:in0"],
    ["TensorArrayV3:out0", "TensorArraySizeV3:in0"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["Switch_3:out0", "Exit_2:in0"], ["Merge:out0", "Switch:in0"],
    ["Identity_2:out4096", "C_9:in0"], ["C_2:out0", "Range:in0"], ["Range:out0", "TensorArrayGatherV3:in1"],
    ["Merge_2:out0", "Switch_1:in0"], ["Switch_4:out1", "Identity_4:in0"],
    ["TensorArrayScatterV3:out0", "Enter_9:in0"],
    ["Transpose_1:out0", "TensorArrayScatterV3:in2"], ["NextIteration:out0", "Merge:in1"],
    ["Enter_10:out0", "Merge_3:in0"],
    ["Identity_2:out4096", "C_13:in0"], ["Identity_3:out0", "Mul_1:in1"], ["C:out0", "Transpose:in1"],
    ["MatMul:out0", "BiasAdd:in0"],
    ["C_1:out0", "TensorArrayV3_1:in0"], ["Add_1:out0", "NextIteration_1:in0"], ["TensorArrayV3:out0", "Enter_1:in0"],
    ["LoopCond:out0", "Switch_1:in1"], ["Enter_8:out0", "TensorArrayReadV3:in0"],
    ["Enter_9:out0", "TensorArrayReadV3:in2"],
    ["Mul:out0", "TensorArrayWriteV3:in2"], ["Identity_2:out4096", "C_8:in0"],
    ["TensorArrayGatherV3:out0", "Transpose:in0"],
    ["Enter_1:out0", "TensorArrayWriteV3:in0"], ["LoopCond:out0", "Switch_3:in1"], ["Enter_2:out0", "Less:in1"],
    ["TensorArrayV3:out1", "Enter:in0"], ["Mul:out0", "NextIteration_4:in0"], ["Merge_1:out0", "Switch_2:in0"]],
"src_in_anchor": [["I:out0", "Transpose_1:in0"], ["I_1:out0", "Enter_11:in0"], ["I_2:out0", "Enter_10:in0"]],
"src_out_tensor": ["Transpose:out0", "Exit_1:out0", "Exit_2:out0"],
"acu_lys_alias": ["reshape_1", "lstmunit", "reshape_2"],
"src_acu_in_tensor_map": [["I:out0", "reshape_1:in0"],
    ["I_1:out0", "lstmunit:in1"], ["I_2:out0", "lstmunit:in2"],
    ],
"src_acu_out_tensor_map": [["Transpose:out0", "reshape_2:out0"],
    ["Exit_1:out0", "lstmunit:out1"], ["Exit_2:out0", "lstmunit:out2"]],
"param_map": {
    "lstmunit": {
        'weights': ['INT', 'PYFUNC', r_lstmunit_rule_get_param_weights(tensor_name='C_12:out0')],
        'num_proj': ['ORIGIN', 'VALUE', None],
        'forget_bias': ['FLOAT', 'PYFUNC', r_lstmunit_rule_get_param_forget_bias(tensor_name='C_13:out0')],
    },
    "reshape_1": {
        'shape': ['ORIGIN', 'PYFUNC', r_lstmunit_rule_lstm_to_lstmunit_reshape1_shape(input_name='I:out0')],
    },
    "reshape_2": {
        'shape': ['ORIGIN', 'PYFUNC', r_lstmunit_rule_lstm_to_lstmunit_reshape2_shape(input_name='I:out0',
            kernel_name='C_12:out0')],
    }},
"blob_map": {"lstmunit": {
    'wi': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_12:out0', input_name='I:out0', index=0)],
    'wc': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_12:out0', input_name='I:out0', index=1)],
    'wf': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_12:out0', input_name='I:out0', index=2)],
    'wo': ['PYFUNC', r_lstmunit_rule_get_weight_w(weight_name='C_12:out0', input_name='I:out0', index=3)],
    'hi': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_12:out0', input_name='I:out0', index=0)],
    'hc': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_12:out0', input_name='I:out0', index=1)],
    'hf': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_12:out0', input_name='I:out0', index=2)],
    'ho': ['PYFUNC', r_lstmunit_rule_get_weight_h(weight_name='C_12:out0', input_name='I:out0', index=3)],
    'bi': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_10:out0', index=0)],
    'bc': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_10:out0', index=1)],
    'bf': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_10:out0', index=2)],
    'bo': ['PYFUNC', r_lstmunit_rule_get_bias(bias_name='C_10:out0', index=3)],
    }},
"acu_inter_flow": [ ["reshape_1:out0", "lstmunit:in0"],["lstmunit:out0", "reshape_2:in0"]],
"priority_tip": 0,
"pre_condition": r_lstmunit_rule_lstm_to_lstmunit_pre_condition(input_name='I:out0') }
ruler_list.append(r_lstmunit_rule_lstm_to_lstmunit)

@rule_pyfunc_def
def r_keras_time_major_lstmlayer_build_kernel(self, node, tensor, tensor_names):
    w_i = self.tensor_to_numpy(tensor[tensor_names[0]]) # i
    w_f = self.tensor_to_numpy(tensor[tensor_names[1]]) # f
    w_j = self.tensor_to_numpy(tensor[tensor_names[2]]) # j
    w_o = self.tensor_to_numpy(tensor[tensor_names[3]]) # o
    h_i = self.tensor_to_numpy(tensor[tensor_names[4]]) # i
    h_f = self.tensor_to_numpy(tensor[tensor_names[5]]) # f
    h_j = self.tensor_to_numpy(tensor[tensor_names[6]]) # j
    h_o = self.tensor_to_numpy(tensor[tensor_names[7]]) # o

    # i j f o
    import numpy as np
    kernel = np.concatenate([w_i, w_j, w_f, w_o], 1)
    recurrent = np.concatenate([h_i, h_j, h_f, h_o], 1)
    data = np.concatenate([kernel, recurrent], 0)
    print("w_i.shape: {}, kernel.shape: {}, recurrent.shape: {}, data.shape: {}".format(
        w_i.shape, kernel.shape, recurrent.shape, data.shape))
    return data

@rule_pyfunc_def
def r_keras_time_major_lstmlayer_build_bias(self, node, tensor, tensor_names):
    b_i = self.tensor_to_numpy(tensor[tensor_names[0]]) # i
    b_f = self.tensor_to_numpy(tensor[tensor_names[1]]) # f
    b_j = self.tensor_to_numpy(tensor[tensor_names[2]]) # j
    b_o = self.tensor_to_numpy(tensor[tensor_names[3]]) # o

    import numpy as np
    bias = np.concatenate([b_i, b_j, b_f, b_o], 0)
    print('b_w.shape: {}, bias.shape: {}'.format(b_i.shape, bias.shape))
    return bias

r_keras_time_major_lstmlayer = {
"ruler_name": "r_keras_time_major_lstmlayer",
"src_ops_alias": ["TensorArrayReadV3", "TensorArrayV3", "Sub", "Exit", "C", "Exit_1", "C_1", "Switch", "Switch_1",
    "Merge", "LoopCond", "Merge_1", "Enter", "NextIteration", "LogicalAnd", "Enter_1",
    "NextIteration_1", "TensorArrayWriteV3", "Less", "Less_1", "C_2", "Add", "Enter_2", "Identity",
    "Mul", "Identity_1", "Merge_2", "Enter_3", "Enter_4", "C_3", "ClipByValue", "Tanh", "Enter_5",
    "NextIteration_2", "C_4", "Identity_2", "Add_1", "C_5", "C_6", "Add_2", "C_7", "Add_3", "Switch_2",
    "Mul_1", "C_8", "Mul_2", "Mul_3", "C_9", "C_10", "Add_4", "ClipByValue_1", "Identity_3",
    "ClipByValue_2", "Tanh_1", "BiasAdd", "MatMul", "Add_5", "C_11", "C_12", "Switch_3", "Add_6",
    "C_13", "C_14", "Add_7", "MatMul_1", "Enter_6", "Identity_4", "Enter_7", "Mul_4", "C_15",
    "Merge_3", "Mul_5", "C_16", "BiasAdd_1", "MatMul_2", "TensorArrayReadV3_1", "Enter_8", "C_17",
    "Switch_4", "C_18", "C_19", "Add_8", "Enter_9", "NextIteration_3", "C_20", "Add_9", "MatMul_3",
    "Enter_10", "Enter_11", "Enter_12", "Enter_13", "C_21", "Merge_4", "BiasAdd_2", "MatMul_4", "C_22",
    "BiasAdd_3", "MatMul_5", "Enter_14", "C_23", "C_24", "TensorArrayV3_1", "TensorArrayScatterV3",
    "Enter_15", "NextIteration_4", "MatMul_6", "Enter_16", "Enter_17", "MatMul_7", "Enter_18",
    "Enter_19", "C_25", "C_26", "C_27", "Enter_20", "C_28", "C_29", "Enter_21", "C_30", "C_31", "C_32",
    "C_33"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayReadV3:in0"], ["Sub:out0", "TensorArrayReadV3:in1"],
    ["Exit:out0", "TensorArrayReadV3:in2"], ["C:out0", "TensorArrayV3:in0"],
    ["Exit_1:out0", "Sub:in0"], ["C_1:out0", "Sub:in1"], ["Switch:out0", "Exit:in0"],
    ["Switch_1:out0", "Exit_1:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Merge_1:out0", "Switch_1:in0"], ["Enter:out0", "Merge:in0"],
    ["NextIteration:out0", "Merge:in1"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["TensorArrayV3:out1", "Enter:in0"], ["Enter_1:out0", "Merge_1:in0"],
    ["NextIteration_1:out0", "Merge_1:in1"], ["TensorArrayWriteV3:out0", "NextIteration:in0"],
    ["Less:out0", "LogicalAnd:in0"], ["Less_1:out0", "LogicalAnd:in1"], ["C_2:out0", "Enter_1:in0"],
    ["Add:out0", "NextIteration_1:in0"], ["Enter_2:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Mul:out0", "TensorArrayWriteV3:in2"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["Merge_1:out0", "Less_1:in0"],
    ["Merge_2:out0", "Less:in0"], ["Enter_3:out0", "Less:in1"], ["Enter_4:out0", "Less_1:in1"],
    ["TensorArrayV3:out0", "Enter_2:in0"], ["Identity:out0", "Add:in0"],
    ["Switch_1:out1", "Identity:in0"], ["C_3:out0", "Add:in1"], ["Switch:out1", "Identity_1:in0"],
    ["ClipByValue:out0", "Mul:in0"], ["Tanh:out0", "Mul:in1"], ["C:out0", "Enter_4:in0"],
    ["Enter_5:out0", "Merge_2:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["C_4:out0", "Enter_3:in0"], ["Identity_2:out4096", "C_3:in0"], ["Add_1:out0", "ClipByValue:in0"],
    ["C_5:out0", "ClipByValue:in1"], ["C_6:out0", "ClipByValue:in2"], ["Add_2:out0", "Tanh:in0"],
    ["C_7:out0", "Enter_5:in0"], ["Add_3:out0", "NextIteration_2:in0"],
    ["Switch_2:out1", "Identity_2:in0"], ["Mul_1:out0", "Add_1:in0"], ["C_8:out0", "Add_1:in1"],
    ["Identity_2:out4096", "C_5:in0"], ["Identity_2:out4096", "C_6:in0"],
    ["LoopCond:out0", "Switch_2:in1"], ["Mul_2:out0", "Add_2:in0"], ["Mul_3:out0", "Add_2:in1"],
    ["Identity_2:out0", "Add_3:in0"], ["Merge_2:out0", "Switch_2:in0"], ["C_9:out0", "Add_3:in1"],
    ["Identity_2:out4096", "C_8:in0"], ["C_10:out0", "Mul_1:in0"], ["Add_4:out0", "Mul_1:in1"],
    ["ClipByValue_1:out0", "Mul_2:in0"], ["Identity_3:out0", "Mul_2:in1"],
    ["Identity_2:out4096", "C_9:in0"], ["ClipByValue_2:out0", "Mul_3:in0"],
    ["Tanh_1:out0", "Mul_3:in1"], ["Identity_2:out4096", "C_10:in0"], ["BiasAdd:out0", "Add_4:in0"],
    ["MatMul:out0", "Add_4:in1"], ["Add_5:out0", "ClipByValue_1:in0"],
    ["C_11:out0", "ClipByValue_1:in1"], ["C_12:out0", "ClipByValue_1:in2"],
    ["Switch_3:out1", "Identity_3:in0"], ["Add_6:out0", "ClipByValue_2:in0"],
    ["C_13:out0", "ClipByValue_2:in1"], ["C_14:out0", "ClipByValue_2:in2"],
    ["Add_7:out0", "Tanh_1:in0"], ["LoopCond:out0", "Switch_3:in1"], ["MatMul_1:out0", "BiasAdd:in0"],
    ["Identity_2:out4096", "C_11:in0"], ["Enter_6:out0", "BiasAdd:in1"],
    ["Identity_2:out4096", "C_12:in0"], ["Identity_4:out0", "MatMul:in0"],
    ["Enter_7:out0", "MatMul:in1"], ["Mul_4:out0", "Add_5:in0"], ["C_15:out0", "Add_5:in1"],
    ["Identity_2:out4096", "C_13:in0"], ["Identity_2:out4096", "C_14:in0"],
    ["Merge_3:out0", "Switch_3:in0"], ["Mul_5:out0", "Add_6:in0"], ["C_16:out0", "Add_6:in1"],
    ["BiasAdd_1:out0", "Add_7:in0"], ["MatMul_2:out0", "Add_7:in1"],
    ["TensorArrayReadV3_1:out0", "MatMul_1:in0"], ["Enter_8:out0", "MatMul_1:in1"],
    ["Identity_2:out4096", "C_15:in0"], ["C_17:out0", "Enter_6:in0"],
    ["Switch_4:out1", "Identity_4:in0"], ["C_18:out0", "Enter_7:in0"],
    ["Identity_2:out4096", "C_16:in0"], ["C_19:out0", "Mul_4:in0"], ["Add_8:out0", "Mul_4:in1"],
    ["Identity:out0", "TensorArrayReadV3_1:in1"], ["Enter_9:out0", "Merge_3:in0"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["LoopCond:out0", "Switch_4:in1"],
    ["C_20:out0", "Mul_5:in0"], ["Add_9:out0", "Mul_5:in1"], ["Identity_4:out0", "MatMul_2:in0"],
    ["MatMul_3:out0", "BiasAdd_1:in0"], ["Enter_10:out0", "BiasAdd_1:in1"],
    ["Enter_11:out0", "MatMul_2:in1"], ["Identity_2:out4096", "C_19:in0"],
    ["Enter_12:out0", "TensorArrayReadV3_1:in0"], ["Enter_13:out0", "TensorArrayReadV3_1:in2"],
    ["C_21:out0", "Enter_8:in0"], ["Add_2:out0", "NextIteration_3:in0"],
    ["Merge_4:out0", "Switch_4:in0"], ["Identity_2:out4096", "C_20:in0"],
    ["BiasAdd_2:out0", "Add_8:in0"], ["MatMul_4:out0", "Add_8:in1"], ["C_22:out0", "Enter_9:in0"],
    ["TensorArrayReadV3_1:out0", "MatMul_3:in0"], ["BiasAdd_3:out0", "Add_9:in0"],
    ["MatMul_5:out0", "Add_9:in1"], ["Enter_14:out0", "MatMul_3:in1"], ["C_23:out0", "Enter_10:in0"],
    ["C_24:out0", "Enter_11:in0"], ["TensorArrayV3_1:out0", "Enter_12:in0"],
    ["TensorArrayScatterV3:out0", "Enter_13:in0"], ["Identity_4:out0", "MatMul_4:in0"],
    ["C:out0", "TensorArrayV3_1:in0"], ["Enter_15:out0", "Merge_4:in0"],
    ["NextIteration_4:out0", "Merge_4:in1"], ["MatMul_6:out0", "BiasAdd_2:in0"],
    ["Enter_16:out0", "BiasAdd_2:in1"], ["Identity_4:out0", "MatMul_5:in0"],
    ["Enter_17:out0", "MatMul_4:in1"], ["Mul:out0", "NextIteration_4:in0"],
    ["MatMul_7:out0", "BiasAdd_3:in0"], ["Enter_18:out0", "BiasAdd_3:in1"],
    ["Enter_19:out0", "MatMul_5:in1"], ["C_25:out0", "Enter_14:in0"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["TensorArrayReadV3_1:out0", "MatMul_6:in0"],
    ["C_26:out0", "TensorArrayScatterV3:in1"], ["C_27:out0", "Enter_15:in0"],
    ["TensorArrayReadV3_1:out0", "MatMul_7:in0"], ["Enter_20:out0", "MatMul_6:in1"],
    ["C_28:out0", "Enter_16:in0"], ["C_29:out0", "Enter_17:in0"], ["Enter_21:out0", "MatMul_7:in1"],
    ["C_30:out0", "Enter_18:in0"], ["C_31:out0", "Enter_19:in0"], ["C_32:out0", "Enter_20:in0"],
    ["C_33:out0", "Enter_21:in0"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["TensorArrayReadV3:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["TensorArrayReadV3:out0", "lstm:out0"]],
"acu_inter_flow": [],
"param_map": {
    "lstm": {
        'time_major': ['BOOL', 'VALUE', True],
        'forget_bias': ['FLOAT', 'VALUE', 0],
        'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_33:out0'])[1]"],
        'recurrent_activation': ['STRING', 'VALUE', 'hard_sigmoid'],
        'return_sequences': ['BOOL', 'VALUE', False],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': [
            'PYFUNC', r_keras_time_major_lstmlayer_build_kernel(tensor_names=[
                'C_33:out0',
                'C_32:out0',
                'C_25:out0',
                'C_21:out0',
                'C_31:out0',
                'C_29:out0',
                'C_24:out0',
                'C_18:out0',
            ])],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': [
            'PYFUNC', r_keras_time_major_lstmlayer_build_bias(tensor_names=[
                'C_30:out0',
                'C_28:out0',
                'C_23:out0',
                'C_17:out0',
            ])],
    }},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_keras_time_major_lstmlayer)

r_keras_time_major_reverse_lstmlayer = {
"ruler_name": "r_keras_time_major_reverse_lstmlayer",
"src_ops_alias": ["TensorArrayReadV3", "TensorArrayV3", "Sub", "Exit", "C", "Exit_1", "C_1", "Switch", "Switch_1",
    "Merge", "LoopCond", "Merge_1", "Enter", "NextIteration", "LogicalAnd", "Enter_1",
    "NextIteration_1", "TensorArrayWriteV3", "Less", "Less_1", "C_2", "Add", "Enter_2", "Identity",
    "Mul", "Identity_1", "Merge_2", "Enter_3", "Enter_4", "C_3", "ClipByValue", "Tanh", "Enter_5",
    "NextIteration_2", "C_4", "Identity_2", "Add_1", "C_5", "C_6", "Add_2", "C_7", "Add_3", "Switch_2",
    "Mul_1", "C_8", "Mul_2", "Mul_3", "C_9", "C_10", "Add_4", "ClipByValue_1", "Identity_3",
    "ClipByValue_2", "Tanh_1", "BiasAdd", "MatMul", "Add_5", "C_11", "C_12", "Switch_3", "Add_6",
    "C_13", "C_14", "Add_7", "MatMul_1", "Enter_6", "Identity_4", "Enter_7", "Mul_4", "C_15",
    "Merge_3", "Mul_5", "C_16", "BiasAdd_1", "MatMul_2", "TensorArrayReadV3_1", "Enter_8", "C_17",
    "Switch_4", "C_18", "C_19", "Add_8", "Enter_9", "NextIteration_3", "C_20", "Add_9", "MatMul_3",
    "Enter_10", "Enter_11", "Enter_12", "Enter_13", "C_21", "Merge_4", "BiasAdd_2", "MatMul_4", "C_22",
    "BiasAdd_3", "MatMul_5", "Enter_14", "C_23", "C_24", "TensorArrayV3_1", "TensorArrayScatterV3",
    "Enter_15", "NextIteration_4", "MatMul_6", "Enter_16", "Enter_17", "MatMul_7", "Enter_18",
    "Enter_19", "C_25", "C_26", "ReverseV2", "C_27", "Enter_20", "C_28", "C_29", "Enter_21", "C_30",
    "C_31", "C_32", "C_33", "C_34"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayReadV3:in0"], ["Sub:out0", "TensorArrayReadV3:in1"],
    ["Exit:out0", "TensorArrayReadV3:in2"], ["C:out0", "TensorArrayV3:in0"],
    ["Exit_1:out0", "Sub:in0"], ["C_1:out0", "Sub:in1"], ["Switch:out0", "Exit:in0"],
    ["Switch_1:out0", "Exit_1:in0"], ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Merge_1:out0", "Switch_1:in0"], ["Enter:out0", "Merge:in0"],
    ["NextIteration:out0", "Merge:in1"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["TensorArrayV3:out1", "Enter:in0"], ["Enter_1:out0", "Merge_1:in0"],
    ["NextIteration_1:out0", "Merge_1:in1"], ["TensorArrayWriteV3:out0", "NextIteration:in0"],
    ["Less:out0", "LogicalAnd:in0"], ["Less_1:out0", "LogicalAnd:in1"], ["C_2:out0", "Enter_1:in0"],
    ["Add:out0", "NextIteration_1:in0"], ["Enter_2:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Mul:out0", "TensorArrayWriteV3:in2"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["Merge_1:out0", "Less_1:in0"],
    ["Merge_2:out0", "Less:in0"], ["Enter_3:out0", "Less:in1"], ["Enter_4:out0", "Less_1:in1"],
    ["TensorArrayV3:out0", "Enter_2:in0"], ["Identity:out0", "Add:in0"],
    ["Switch_1:out1", "Identity:in0"], ["C_3:out0", "Add:in1"], ["Switch:out1", "Identity_1:in0"],
    ["ClipByValue:out0", "Mul:in0"], ["Tanh:out0", "Mul:in1"], ["C:out0", "Enter_4:in0"],
    ["Enter_5:out0", "Merge_2:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["C_4:out0", "Enter_3:in0"], ["Identity_2:out4096", "C_3:in0"], ["Add_1:out0", "ClipByValue:in0"],
    ["C_5:out0", "ClipByValue:in1"], ["C_6:out0", "ClipByValue:in2"], ["Add_2:out0", "Tanh:in0"],
    ["C_7:out0", "Enter_5:in0"], ["Add_3:out0", "NextIteration_2:in0"],
    ["Switch_2:out1", "Identity_2:in0"], ["Mul_1:out0", "Add_1:in0"], ["C_8:out0", "Add_1:in1"],
    ["Identity_2:out4096", "C_5:in0"], ["Identity_2:out4096", "C_6:in0"],
    ["LoopCond:out0", "Switch_2:in1"], ["Mul_2:out0", "Add_2:in0"], ["Mul_3:out0", "Add_2:in1"],
    ["Identity_2:out0", "Add_3:in0"], ["Merge_2:out0", "Switch_2:in0"], ["C_9:out0", "Add_3:in1"],
    ["Identity_2:out4096", "C_8:in0"], ["C_10:out0", "Mul_1:in0"], ["Add_4:out0", "Mul_1:in1"],
    ["ClipByValue_1:out0", "Mul_2:in0"], ["Identity_3:out0", "Mul_2:in1"],
    ["Identity_2:out4096", "C_9:in0"], ["ClipByValue_2:out0", "Mul_3:in0"],
    ["Tanh_1:out0", "Mul_3:in1"], ["Identity_2:out4096", "C_10:in0"], ["BiasAdd:out0", "Add_4:in0"],
    ["MatMul:out0", "Add_4:in1"], ["Add_5:out0", "ClipByValue_1:in0"],
    ["C_11:out0", "ClipByValue_1:in1"], ["C_12:out0", "ClipByValue_1:in2"],
    ["Switch_3:out1", "Identity_3:in0"], ["Add_6:out0", "ClipByValue_2:in0"],
    ["C_13:out0", "ClipByValue_2:in1"], ["C_14:out0", "ClipByValue_2:in2"],
    ["Add_7:out0", "Tanh_1:in0"], ["LoopCond:out0", "Switch_3:in1"], ["MatMul_1:out0", "BiasAdd:in0"],
    ["Identity_2:out4096", "C_11:in0"], ["Enter_6:out0", "BiasAdd:in1"],
    ["Identity_2:out4096", "C_12:in0"], ["Identity_4:out0", "MatMul:in0"],
    ["Enter_7:out0", "MatMul:in1"], ["Mul_4:out0", "Add_5:in0"], ["C_15:out0", "Add_5:in1"],
    ["Identity_2:out4096", "C_13:in0"], ["Identity_2:out4096", "C_14:in0"],
    ["Merge_3:out0", "Switch_3:in0"], ["Mul_5:out0", "Add_6:in0"], ["C_16:out0", "Add_6:in1"],
    ["BiasAdd_1:out0", "Add_7:in0"], ["MatMul_2:out0", "Add_7:in1"],
    ["TensorArrayReadV3_1:out0", "MatMul_1:in0"], ["Enter_8:out0", "MatMul_1:in1"],
    ["Identity_2:out4096", "C_15:in0"], ["C_17:out0", "Enter_6:in0"],
    ["Switch_4:out1", "Identity_4:in0"], ["C_18:out0", "Enter_7:in0"],
    ["Identity_2:out4096", "C_16:in0"], ["C_19:out0", "Mul_4:in0"], ["Add_8:out0", "Mul_4:in1"],
    ["Identity:out0", "TensorArrayReadV3_1:in1"], ["Enter_9:out0", "Merge_3:in0"],
    ["NextIteration_3:out0", "Merge_3:in1"], ["LoopCond:out0", "Switch_4:in1"],
    ["C_20:out0", "Mul_5:in0"], ["Add_9:out0", "Mul_5:in1"], ["Identity_4:out0", "MatMul_2:in0"],
    ["MatMul_3:out0", "BiasAdd_1:in0"], ["Enter_10:out0", "BiasAdd_1:in1"],
    ["Enter_11:out0", "MatMul_2:in1"], ["Identity_2:out4096", "C_19:in0"],
    ["Enter_12:out0", "TensorArrayReadV3_1:in0"], ["Enter_13:out0", "TensorArrayReadV3_1:in2"],
    ["C_21:out0", "Enter_8:in0"], ["Add_2:out0", "NextIteration_3:in0"],
    ["Merge_4:out0", "Switch_4:in0"], ["Identity_2:out4096", "C_20:in0"],
    ["BiasAdd_2:out0", "Add_8:in0"], ["MatMul_4:out0", "Add_8:in1"], ["C_22:out0", "Enter_9:in0"],
    ["TensorArrayReadV3_1:out0", "MatMul_3:in0"], ["BiasAdd_3:out0", "Add_9:in0"],
    ["MatMul_5:out0", "Add_9:in1"], ["Enter_14:out0", "MatMul_3:in1"], ["C_23:out0", "Enter_10:in0"],
    ["C_24:out0", "Enter_11:in0"], ["TensorArrayV3_1:out0", "Enter_12:in0"],
    ["TensorArrayScatterV3:out0", "Enter_13:in0"], ["Identity_4:out0", "MatMul_4:in0"],
    ["C:out0", "TensorArrayV3_1:in0"], ["Enter_15:out0", "Merge_4:in0"],
    ["NextIteration_4:out0", "Merge_4:in1"], ["MatMul_6:out0", "BiasAdd_2:in0"],
    ["Identity_4:out0", "MatMul_5:in0"], ["Enter_16:out0", "BiasAdd_2:in1"],
    ["Enter_17:out0", "MatMul_4:in1"], ["Mul:out0", "NextIteration_4:in0"],
    ["MatMul_7:out0", "BiasAdd_3:in0"], ["Enter_18:out0", "BiasAdd_3:in1"],
    ["Enter_19:out0", "MatMul_5:in1"], ["C_25:out0", "Enter_14:in0"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["TensorArrayReadV3_1:out0", "MatMul_6:in0"],
    ["C_26:out0", "TensorArrayScatterV3:in1"], ["ReverseV2:out0", "TensorArrayScatterV3:in2"],
    ["C_27:out0", "Enter_15:in0"], ["TensorArrayReadV3_1:out0", "MatMul_7:in0"],
    ["Enter_20:out0", "MatMul_6:in1"], ["C_28:out0", "Enter_16:in0"], ["C_29:out0", "Enter_17:in0"],
    ["Enter_21:out0", "MatMul_7:in1"], ["C_30:out0", "Enter_18:in0"], ["C_31:out0", "Enter_19:in0"],
    ["C_32:out0", "ReverseV2:in1"], ["C_33:out0", "Enter_20:in0"], ["C_34:out0", "Enter_21:in0"]],
"src_in_anchor": [["I:out0", "ReverseV2:in0"]],
"src_out_tensor": ["TensorArrayReadV3:out0"],
"acu_lys_alias": ["reverse", "lstm"],
"src_acu_in_tensor_map": [["I:out0", "reverse:in0"]],
"src_acu_out_tensor_map": [["TensorArrayReadV3:out0", "lstm:out0"]],
"acu_inter_flow": [["reverse:out0", "lstm:in0"]],
"param_map": {
    "lstm": {
        'time_major': ['BOOL', 'VALUE', True],
        'forget_bias': ['FLOAT', 'VALUE', 0],
        'weights': ['INT', 'CODE', "self.shape_pick(tensor['C_34:out0'])[1]"],
        'recurrent_activation': ['STRING', 'VALUE', 'hard_sigmoid'],
        'return_sequences': ['BOOL', 'VALUE', False],
    },
    "reverse": {
        'axis':['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_32:out0'])"],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': [
            'PYFUNC', r_keras_time_major_lstmlayer_build_kernel(tensor_names=[
                'C_34:out0',
                'C_33:out0',
                'C_25:out0',
                'C_21:out0',
                'C_31:out0',
                'C_29:out0',
                'C_24:out0',
                'C_18:out0',
            ])],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': [
            'PYFUNC', r_keras_time_major_lstmlayer_build_bias(tensor_names=[
                'C_30:out0',
                'C_28:out0',
                'C_23:out0',
                'C_17:out0',
            ])],
    }},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_keras_time_major_reverse_lstmlayer)


@rule_pyfunc_def
def r_reduce_sum_get_axis(self, node, tensor, axis_tensor_name):
    axis = self.tensor_to_numpy(tensor[axis_tensor_name]).tolist()
    if isinstance(axis, list):
        return axis
    else:
        return [axis]

r_reduce_sum = {
"ruler_name": "reducesum",
"src_ops_alias": ["Sum", "C"],
"src_inter_flow": [["C:out0", "Sum:in1"]],
"src_in_anchor": [["I:out0", "Sum:in0"]],
"src_out_tensor": ["Sum:out0"],
"acu_lys_alias": ["reducesum"],
"src_acu_in_tensor_map": [["I:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["Sum:out0", "reducesum:out0"]],
"param_map": {"reducesum": {
    'axis_list': ['ORIGIN', 'PYFUNC', r_reduce_sum_get_axis(axis_tensor_name='C:out0')],
    'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['Sum'], 'keep_dims')"]
    }},
"blob_map": {"reducesum": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_reduce_sum)

@rule_pyfunc_def
def r_reducesum_output_scalar_pre_condition(self, node, tensor, input_tensor_name):
    input_tensor_shape = self.shape_pick(tensor[input_tensor_name])
    if len(input_tensor_shape) == 1:
        print(input_tensor_shape)
        return True
    return False

r_reducesum_output_scalar = {
"ruler_name": "reducesum_output_scalar",
"src_ops_alias": ["Pack", "Sum", "C"],
"src_inter_flow": [["Sum:out0", "Pack:in0"], ["C:out0", "Sum:in1"]],
"src_in_anchor": [["I:out0", "Sum:in0"]],
"src_out_tensor": ["Pack:out0"],
"acu_lys_alias": ["reducesum"],
"src_acu_in_tensor_map": [["I:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["Pack:out0", "reducesum:out0"]],
"acu_inter_flow": [],
"param_map": {"reducesum": {
    'axis_list': ['INTS', 'VALUE', [0]],
    'keep_dims': ['BOOL', 'VALUE', True]
}},
    "blob_map": {"reducesum": {}},
"priority_tip": 0,
"pre_condition": r_reducesum_output_scalar_pre_condition(input_tensor_name='I:out0')}
ruler_list.append(r_reducesum_output_scalar)

r_reduce_any = {
    "ruler_name": "reduceany",
    "src_ops_alias": ["Any", "C"],
    "src_inter_flow": [["C:out0", "Any:in1"]],
    "src_in_anchor": [["I:out0", "Any:in0"]],
    "src_out_tensor": ["Any:out0"],
    "acu_lys_alias": ["reduceany"],
    "src_acu_in_tensor_map": [["I:out0", "reduceany:in0"]],
    "src_acu_out_tensor_map": [["Any:out0", "reduceany:out0"]],
    "param_map": {"reduceany": {
        'axis_list': ['ORIGIN', 'PYFUNC', r_reduce_sum_get_axis(axis_tensor_name='C:out0')],
        'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['Any'], 'keep_dims')"]
    }},
    "blob_map": {"reduceany": {}},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None }
ruler_list.append(r_reduce_any)

r_nms_v5 = {
    "ruler_name": "nms_v5",
    "src_ops_alias": ["NonMaxSuppressionV5", "C", "C_1", "C_2", "C_3"],
    "src_inter_flow": [["C:out0", "NonMaxSuppressionV5:in2"], ["C_1:out0", "NonMaxSuppressionV5:in3"],
        ["C_2:out0", "NonMaxSuppressionV5:in4"], ["C_3:out0", "NonMaxSuppressionV5:in5"]],
    "src_in_anchor": [["I:out0", "NonMaxSuppressionV5:in0"], ["I_1:out0", "NonMaxSuppressionV5:in1"]],
    "src_out_tensor": ["NonMaxSuppressionV5:out0", "NonMaxSuppressionV5:out1", "NonMaxSuppressionV5:out2"],
    "acu_lys_alias": ["nms"],
    "src_acu_in_tensor_map": [["I:out0", "nms:in0"], ["I_1:out0", "nms:in1"]],
    "src_acu_out_tensor_map": [["NonMaxSuppressionV5:out0", "nms:out0"], ["NonMaxSuppressionV5:out1", "nms:out1"],
                               ["NonMaxSuppressionV5:out2", "nms:out2"]],
    "acu_inter_flow": [],
    "param_map": {"nms": {'max_output_size': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])[0]"],
                          'iou_threshold': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])[0]"],
                          'score_threshold': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])[0]"],
                          'soft_nms_sigma': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_3:out0'])[0]"],
                          'pad_to_max_output_size':
                              ['BOOL', 'CODE', "self.attr_pick(node['NonMaxSuppressionV5'],'pad_to_max_output_size')"]
                          },
                  },
    "blob_map": {"nms": {}},
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_nms_v5)

r_swish = {
"ruler_name": "swish",
"src_ops_alias": ["swish_f32"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "swish_f32:in0"]],
"src_out_tensor": ["swish_f32:out0"],
"acu_lys_alias": ["swish"],
"src_acu_in_tensor_map": [["I:out0", "swish:in0"]],
"src_acu_out_tensor_map": [["swish_f32:out0", "swish:out0"]],
"acu_inter_flow": [],
"param_map": {"swish": {}},
"blob_map": {"swish": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_swish)

r_swish_tf23 = {
    "ruler_name": "swish_tf23",
    "src_ops_alias": ["Sigmoid", "Mul", "IdentityN"],
    "src_inter_flow": [["Sigmoid:out0", "Mul:in1"], ["Mul:out0", "IdentityN:in0"]],
    "src_in_anchor": [["I:out0", "Sigmoid:in0"], ["I:out0", "Mul:in0"], ["I:out0", "IdentityN:in1"]],
    "src_out_tensor": ["IdentityN:out0"],
    "acu_lys_alias": ["swish"],
    "src_acu_in_tensor_map": [["I:out0", "swish:in0"]],
    "src_acu_out_tensor_map": [["IdentityN:out0", "swish:out0"]],
    "acu_inter_flow": [],
    "param_map": {"swish": {}},
    "blob_map": {"swish": {}},
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_swish_tf23)

@rule_pyfunc_def
def r_signal_frame_get_frame_len(self, node, tensor, paddings_tensor_name, output_tensor_name):
    axis = None
    paddings = self.tensor_to_numpy(tensor[paddings_tensor_name])
    for i, pad in enumerate(paddings):
        if (pad != [0, 0]).any():
            if axis is None:
                axis = i
            else:
                raise ValueError('Cannot determine axis')
    if axis is None:
        raise ValueError('Cannot determine axis')

    output_tensor_shape = self.shape_pick(tensor[output_tensor_name])
    return output_tensor_shape[axis + 1]

@rule_pyfunc_def
def r_signal_frame_get_frame_step(self, node, tensor, paddings_tensor_name, input_tensor_name,
                            output_tensor_name):
    axis = None
    pad_value = None
    paddings = self.tensor_to_numpy(tensor[paddings_tensor_name])
    for i, pad in enumerate(paddings):
        if (pad != [0, 0]).any():
            if axis is None:
                axis = i
                pad_value = pad[1]
            else:
                raise ValueError('Cannot determine axis')
    if axis is None or pad_value is None:
        raise ValueError('Cannot determine axis')

    input_tensor_shape = self.shape_pick(tensor[input_tensor_name])
    output_tensor_shape = self.shape_pick(tensor[output_tensor_name])
    output_len = output_tensor_shape[axis]
    frame_length = output_tensor_shape[axis + 1]
    if (input_tensor_shape[axis] + pad_value - frame_length) == 0:
        frame_step = 1
    else:
        frame_step = (input_tensor_shape[axis] + pad_value - frame_length) / (output_len - 1)

    return int(frame_step)

@rule_pyfunc_def
def r_signal_frame_get_axis(self, node, tensor, paddings_tensor_name):
    axis = None
    paddings = self.tensor_to_numpy(tensor[paddings_tensor_name])
    for i, pad in enumerate(paddings):
        if (pad != [0, 0]).any():
            if axis is None:
                axis = i
            else:
                raise ValueError('Cannot determine axis')
    if axis is None:
        raise ValueError('Cannot determine axis')

    return axis

r_signal_frame = {
"ruler_name": "frame",
"src_ops_alias": ["Reshape", "GatherV2", "C", "Reshape_1", "C_1", "C_2", "StridedSlice",
    "C_3", "PadV2", "C_4", "C_5", "C_6", "C_7", "C_8"],
"src_inter_flow": [["C:out0", "Reshape:in1"], ["C_6:out0", "StridedSlice:in3"],
    ["PadV2:out0", "StridedSlice:in0"], ["StridedSlice:out0", "Reshape_1:in0"],
    ["C_2:out0", "GatherV2:in2"], ["C_4:out0", "StridedSlice:in1"], ["C_1:out0", "GatherV2:in1"],
    ["C_3:out0", "Reshape_1:in1"], ["Reshape_1:out0", "GatherV2:in0"], ["C_7:out0", "PadV2:in1"],
    ["C_5:out0", "StridedSlice:in2"], ["C_8:out0", "PadV2:in2"], ["GatherV2:out0", "Reshape:in0"]],
"src_in_anchor": [["I:out0", "PadV2:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["signalframe"],
"src_acu_in_tensor_map": [["I:out0", "signalframe:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "signalframe:out0"]],
"param_map": {"signalframe": {
    'frame_length': ['ORIGIN', 'PYFUNC', r_signal_frame_get_frame_len(paddings_tensor_name='C_7:out0',
        output_tensor_name='Reshape:out0')],
    'frame_step': ['ORIGIN', 'PYFUNC', r_signal_frame_get_frame_step(paddings_tensor_name='C_7:out0',
        input_tensor_name='I:out0',
        output_tensor_name='Reshape:out0')],
    'pad_end': ['BOOL', 'VALUE', True],
    'pad_value': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_8:out0'])"],
    'axis': ['ORIGIN', 'PYFUNC', r_signal_frame_get_axis(paddings_tensor_name='C_7:out0')],
    }},
"blob_map": {"signalframe": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None }
ruler_list.append(r_signal_frame)

# r_last = dict()
# r_last['src_ops_alias'] = ['customer_slot_1']
# r_last['acu_lys_alias'] = ['customer_slot_1']
# r_last['src_acu_inputs_map'] = ['CODE', "[[['customer_slot_1', 'in'+str(port)],
#    ['customer_slot_1', 'in'+str(port)]] for port in range(len(tensor['customer_slot_1'].input))]"]
# r_last['src_acu_outputs_map'] = ['CODE', "[[['customer_slot_1', 'out'+str(port)],
#    ['customer_slot_1', 'out'+str(port)]] for port in range(len(tensor['customer_slot_1'].input))]"]
# r_last['priority_tip'] = -9
# ruler_list.append(r_last)

r_mrcnn_proposal = {
"ruler_name": "mrcnn-proposal",
"src_ops_alias": ["Pack", "Pad", "GatherV2", "C", "StridedSlice", "NonMaxSuppressionV3", "C_1", "Pack_1", "C_2",
    "C_3", "C_4", "StridedSlice_1", "C_5", "C_6", "C_7", "ConcatV2", "Pack_2", "C_8", "C_9", "C_10",
    "Maximum", "Maximum_1", "Maximum_2", "Maximum_3", "C_11", "GatherV2_1", "Minimum", "C_12",
    "Minimum_1", "C_13", "Minimum_2", "Minimum_3", "StridedSlice_2", "StridedSlice_3", "C_14", "Split",
    "C_15", "C_16", "StridedSlice_4", "C_17", "C_18", "C_19", "TopKV2", "C_20", "C_21", "C_22", "C_23",
    "StridedSlice_5", "C_24", "C_25", "C_26", "C_27", "Pack_3", "C_28", "C_29", "C_30", "Pack_4",
    "Sub", "Sub_1", "Add", "Add_1", "Add_2", "Mul", "Add_3", "Mul_1", "Mul_2", "Mul_3", "Add_4",
    "Mul_4", "C_31", "Add_5", "Mul_5", "C_32", "Sub_2", "Exp", "Sub_3", "Exp_1", "StridedSlice_6",
    "Mul_6", "StridedSlice_7", "StridedSlice_8", "Mul_7", "StridedSlice_9", "StridedSlice_10",
    "StridedSlice_11", "StridedSlice_12", "StridedSlice_13", "StridedSlice_14", "StridedSlice_15",
    "StridedSlice_16", "C_33", "C_34", "C_35", "C_36", "StridedSlice_17", "C_37", "C_38", "C_39",
    "C_40", "C_41", "C_42", "C_43", "C_44", "C_45", "C_46", "C_47", "C_48", "C_49", "C_50", "C_51",
    "C_52", "C_53", "C_54", "C_55", "C_56", "C_57", "C_58", "C_59", "C_60", "C_61", "C_62", "C_63",
    "C_64", "Pack_5", "C_65", "C_66", "C_67", "Pack_6", "C_68", "C_69", "C_70", "GatherV2_2",
    "GatherV2_3", "StridedSlice_18", "StridedSlice_19", "C_71", "StridedSlice_20", "StridedSlice_21",
    "C_72", "C_73", "C_74", "C_75", "C_76", "C_77", "C_78", "Mul_8", "C_79", "C_80", "C_81", "C_82",
    "C_83", "C_84", "C_85"],
"src_inter_flow": [["Pad:out0", "Pack:in0"], ["GatherV2:out0", "Pad:in0"], ["C:out0", "Pad:in1"],
    ["StridedSlice:out0", "GatherV2:in0"], ["NonMaxSuppressionV3:out0", "GatherV2:in1"],
    ["C_1:out0", "GatherV2:in2"], ["Pack_1:out0", "StridedSlice:in0"],
    ["C_2:out0", "StridedSlice:in1"], ["C_3:out0", "StridedSlice:in2"],
    ["C_4:out0", "StridedSlice:in3"], ["StridedSlice:out0", "NonMaxSuppressionV3:in0"],
    ["StridedSlice_1:out0", "NonMaxSuppressionV3:in1"], ["C_5:out0", "NonMaxSuppressionV3:in2"],
    ["C_6:out0", "NonMaxSuppressionV3:in3"], ["C_7:out0", "NonMaxSuppressionV3:in4"],
    ["ConcatV2:out0", "Pack_1:in0"], ["Pack_2:out0", "StridedSlice_1:in0"],
    ["C_8:out0", "StridedSlice_1:in1"], ["C_9:out0", "StridedSlice_1:in2"],
    ["C_10:out0", "StridedSlice_1:in3"], ["Maximum:out0", "ConcatV2:in0"],
    ["Maximum_1:out0", "ConcatV2:in1"], ["Maximum_2:out0", "ConcatV2:in2"],
    ["Maximum_3:out0", "ConcatV2:in3"], ["C_11:out0", "ConcatV2:in4"],
    ["GatherV2_1:out0", "Pack_2:in0"], ["Minimum:out0", "Maximum:in0"], ["C_12:out0", "Maximum:in1"],
    ["Minimum_1:out0", "Maximum_1:in0"], ["C_13:out0", "Maximum_1:in1"],
    ["C_12:out0", "Maximum_2:in1"], ["Minimum_2:out0", "Maximum_2:in0"],
    ["C_13:out0", "Maximum_3:in1"], ["Minimum_3:out0", "Maximum_3:in0"],
    ["StridedSlice_2:out0", "GatherV2_1:in0"], ["StridedSlice_3:out0", "GatherV2_1:in1"],
    ["C_14:out0", "GatherV2_1:in2"], ["Split:out0", "Minimum:in0"], ["C_15:out0", "Minimum:in1"],
    ["Split:out1", "Minimum_1:in0"], ["C_16:out0", "Minimum_1:in1"], ["Split:out2", "Minimum_2:in0"],
    ["C_15:out0", "Minimum_2:in1"], ["Split:out3", "Minimum_3:in0"], ["C_16:out0", "Minimum_3:in1"],
    ["StridedSlice_4:out0", "StridedSlice_2:in0"], ["C_17:out0", "StridedSlice_2:in1"],
    ["C_18:out0", "StridedSlice_2:in2"], ["C_19:out0", "StridedSlice_2:in3"],
    ["TopKV2:out1", "StridedSlice_3:in0"], ["C_20:out0", "StridedSlice_3:in1"],
    ["C_21:out0", "StridedSlice_3:in2"], ["C_22:out0", "StridedSlice_3:in3"],
    ["C_23:out0", "Split:in0"], ["StridedSlice_5:out0", "Split:in1"],
    ["C_24:out0", "StridedSlice_4:in1"], ["C_25:out0", "StridedSlice_4:in2"],
    ["C_26:out0", "StridedSlice_4:in3"], ["StridedSlice_4:out0", "TopKV2:in0"],
    ["C_27:out0", "TopKV2:in1"], ["Pack_3:out0", "StridedSlice_5:in0"],
    ["C_28:out0", "StridedSlice_5:in1"], ["C_29:out0", "StridedSlice_5:in2"],
    ["C_30:out0", "StridedSlice_5:in3"], ["Pack_4:out0", "Pack_3:in0"], ["Sub:out0", "Pack_4:in0"],
    ["Sub_1:out0", "Pack_4:in1"], ["Add:out0", "Pack_4:in2"], ["Add_1:out0", "Pack_4:in3"],
    ["Add_2:out0", "Sub:in0"], ["Mul:out0", "Sub:in1"], ["Add_3:out0", "Sub_1:in0"],
    ["Mul_1:out0", "Sub_1:in1"], ["Sub:out0", "Add:in0"], ["Mul_2:out0", "Add:in1"],
    ["Sub_1:out0", "Add_1:in0"], ["Mul_3:out0", "Add_1:in1"], ["Add_4:out0", "Add_2:in0"],
    ["Mul_4:out0", "Add_2:in1"], ["Mul_2:out0", "Mul:in1"], ["C_31:out0", "Mul:in0"],
    ["Add_5:out0", "Add_3:in0"], ["Mul_5:out0", "Add_3:in1"], ["Mul_3:out0", "Mul_1:in1"],
    ["C_32:out0", "Mul_1:in0"], ["Sub_2:out0", "Mul_2:in0"], ["Exp:out0", "Mul_2:in1"],
    ["Sub_3:out0", "Mul_3:in0"], ["Exp_1:out0", "Mul_3:in1"], ["StridedSlice_6:out0", "Add_4:in0"],
    ["Mul_6:out0", "Add_4:in1"], ["Sub_2:out0", "Mul_4:in1"], ["StridedSlice_7:out0", "Mul_4:in0"],
    ["StridedSlice_8:out0", "Add_5:in0"], ["Mul_7:out0", "Add_5:in1"], ["Sub_3:out0", "Mul_5:in1"],
    ["StridedSlice_9:out0", "Mul_5:in0"], ["StridedSlice_10:out0", "Sub_2:in0"],
    ["StridedSlice_11:out0", "Sub_2:in1"], ["StridedSlice_12:out0", "Exp:in0"],
    ["StridedSlice_13:out0", "Sub_3:in0"], ["StridedSlice_14:out0", "Sub_3:in1"],
    ["StridedSlice_15:out0", "Exp_1:in0"], ["Sub_2:out0", "Mul_6:in1"],
    ["StridedSlice_16:out0", "StridedSlice_6:in0"], ["C_33:out0", "StridedSlice_6:in1"],
    ["C_34:out0", "StridedSlice_6:in2"], ["C_35:out0", "StridedSlice_6:in3"],
    ["C_36:out0", "Mul_6:in0"], ["StridedSlice_17:out0", "StridedSlice_7:in0"],
    ["Sub_3:out0", "Mul_7:in1"], ["C_37:out0", "StridedSlice_7:in1"],
    ["C_38:out0", "StridedSlice_7:in2"], ["C_39:out0", "StridedSlice_7:in3"],
    ["StridedSlice_16:out0", "StridedSlice_8:in0"], ["C_40:out0", "StridedSlice_8:in1"],
    ["C_41:out0", "StridedSlice_8:in2"], ["C_42:out0", "StridedSlice_8:in3"],
    ["C_43:out0", "Mul_7:in0"], ["StridedSlice_17:out0", "StridedSlice_9:in0"],
    ["StridedSlice_16:out0", "StridedSlice_10:in0"], ["C_44:out0", "StridedSlice_9:in1"],
    ["C_45:out0", "StridedSlice_9:in2"], ["C_46:out0", "StridedSlice_9:in3"],
    ["StridedSlice_16:out0", "StridedSlice_11:in0"], ["C_47:out0", "StridedSlice_10:in1"],
    ["C_48:out0", "StridedSlice_10:in2"], ["C_49:out0", "StridedSlice_10:in3"],
    ["StridedSlice_17:out0", "StridedSlice_12:in0"], ["C_50:out0", "StridedSlice_11:in1"],
    ["StridedSlice_16:out0", "StridedSlice_13:in0"], ["C_51:out0", "StridedSlice_11:in2"],
    ["C_52:out0", "StridedSlice_11:in3"], ["StridedSlice_16:out0", "StridedSlice_14:in0"],
    ["C_53:out0", "StridedSlice_12:in1"], ["C_54:out0", "StridedSlice_12:in2"],
    ["C_55:out0", "StridedSlice_12:in3"], ["C_56:out0", "StridedSlice_13:in1"],
    ["StridedSlice_17:out0", "StridedSlice_15:in0"], ["C_57:out0", "StridedSlice_13:in2"],
    ["C_58:out0", "StridedSlice_13:in3"], ["C_59:out0", "StridedSlice_14:in1"],
    ["C_60:out0", "StridedSlice_14:in2"], ["C_61:out0", "StridedSlice_14:in3"],
    ["C_62:out0", "StridedSlice_15:in1"], ["C_63:out0", "StridedSlice_15:in2"],
    ["C_64:out0", "StridedSlice_15:in3"], ["Pack_5:out0", "StridedSlice_16:in0"],
    ["C_65:out0", "StridedSlice_16:in1"], ["C_66:out0", "StridedSlice_16:in2"],
    ["C_67:out0", "StridedSlice_16:in3"], ["Pack_6:out0", "StridedSlice_17:in0"],
    ["C_68:out0", "StridedSlice_17:in1"], ["C_69:out0", "StridedSlice_17:in2"],
    ["C_70:out0", "StridedSlice_17:in3"], ["GatherV2_2:out0", "Pack_5:in0"],
    ["TopKV2:out1", "StridedSlice_19:in0"], ["GatherV2_3:out0", "Pack_6:in0"],
    ["TopKV2:out1", "StridedSlice_21:in0"], ["StridedSlice_18:out0", "GatherV2_2:in0"],
    ["StridedSlice_19:out0", "GatherV2_2:in1"], ["C_71:out0", "GatherV2_2:in2"],
    ["StridedSlice_20:out0", "GatherV2_3:in0"], ["StridedSlice_21:out0", "GatherV2_3:in1"],
    ["C_72:out0", "GatherV2_3:in2"], ["C_73:out0", "StridedSlice_18:in1"],
    ["C_74:out0", "StridedSlice_18:in2"], ["C_75:out0", "StridedSlice_18:in3"],
    ["C_76:out0", "StridedSlice_19:in1"], ["C_77:out0", "StridedSlice_19:in2"],
    ["C_78:out0", "StridedSlice_19:in3"], ["Mul_8:out0", "StridedSlice_20:in0"],
    ["C_79:out0", "StridedSlice_20:in1"], ["C_80:out0", "StridedSlice_20:in2"],
    ["C_81:out0", "StridedSlice_20:in3"], ["C_82:out0", "StridedSlice_21:in1"],
    ["C_83:out0", "StridedSlice_21:in2"], ["C_84:out0", "StridedSlice_21:in3"],
    ["C_85:out0", "Mul_8:in1"]],
"src_in_anchor": [["I:out0", "StridedSlice_4:in0"], ["I_1:out0", "Mul_8:in0"], ["I_2:out0", "StridedSlice_18:in0"]],
"src_out_tensor": ["Pack:out0"],
"acu_lys_alias": ["mrcnn_proposal"],
"src_acu_in_tensor_map": [["I:out0", "mrcnn_proposal:in0"], ["I_1:out0", "mrcnn_proposal:in1"],
                          ["I_2:out0", "mrcnn_proposal:in2"]],
"src_acu_out_tensor_map": [["Pack:out0", "mrcnn_proposal:out0"]],
"acu_inter_flow": [],
"param_map": {"mrcnn_proposal": {}},
"blob_map": {"mrcnn_proposal": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mrcnn_proposal)

r_mrcnn_detection = {
"ruler_name": "mrcnn-detection",
"src_ops_alias": ["Reshape", "Pack", "C", "Pad", "ConcatV2", "C_1", "GatherV2", "StridedSlice", "StridedSlice_1",
    "C_2", "ConcatV2_1", "GatherV2_1", "C_3", "Cast", "C_4", "C_5", "C_6", "GatherV2_2", "C_7", "C_8", "C_9",
    "Maximum", "Maximum_1", "Maximum_2", "Maximum_3", "C_10", "StridedSlice_2", "TopKV2", "C_11",
    "GatherV2_3", "GatherNd", "C_12", "Minimum", "Split", "Minimum_1", "Minimum_2", "Minimum_3",
    "SparseToDense", "C_13", "C_14", "C_15", "GatherV2_4", "C_16", "ArgMax", "C_17", "StridedSlice_3",
    "Pack_1", "Split_1", "C_18", "StridedSlice_4", "DenseToDenseSetOperation", "C_19", "C_20", "C_21",
    "C_22", "C_23", "C_24", "C_25", "C_26", "Pack_2", "RealDiv", "C_27", "C_28", "C_29", "ExpandDims",
    "ExpandDims_1", "Sub", "Sub_1", "Add", "Add_1", "Sub_2", "Sub_3", "StridedSlice_5", "C_30",
    "GatherV2_5", "C_31", "Add_2", "Mul", "Add_3", "Mul_1", "Mul_2", "Mul_3", "StridedSlice_6", "C_32",
    "ConcatV2_2", "C_33", "Where", "C_34", "C_35", "C_36", "Reshape_1", "StridedSlice_7", "C_37",
    "Add_4", "Mul_4", "C_38", "Add_5", "Mul_5", "C_39", "Sub_4", "Exp", "Sub_5", "Exp_1", "C_40",
    "C_41", "C_42", "Split_2", "C_43", "Greater", "TensorArrayGatherV3", "C_44", "Where_1", "C_45",
    "C_46", "C_47", "StridedSlice_8", "Mul_6", "StridedSlice_9", "StridedSlice_10", "Mul_7",
    "StridedSlice_11", "StridedSlice_12", "StridedSlice_13", "StridedSlice_14", "StridedSlice_15",
    "StridedSlice_16", "StridedSlice_17", "C_48", "StridedSlice_18", "C_49", "TensorArrayV3", "Range",
    "Exit", "Greater_1", "StridedSlice_19", "C_50", "C_51", "C_52", "C_53", "Mul_8", "C_54", "C_55",
    "C_56", "C_57", "C_58", "C_59", "C_60", "C_61", "C_62", "C_63", "C_64", "C_65", "C_66", "C_67",
    "C_68", "C_69", "C_70", "C_71", "C_72", "C_73", "C_74", "C_75", "C_76", "C_77", "C_78", "C_79",
    "C_80", "C_81", "StridedSlice_20", "C_82", "C_83", "C_84", "C_85", "C_86", "TensorArraySizeV3",
    "C_87", "Switch", "C_88", "C_89", "C_90", "C_91", "GatherNd_1", "C_92", "StridedSlice_21", "C_93",
    "C_94", "C_95", "Merge", "LoopCond", "StridedSlice_22", "C_96", "C_97", "C_98", "Enter",
    "NextIteration", "LogicalAnd", "C_99", "C_100", "C_101", "TensorArrayWriteV3", "Less", "Less_1",
    "Enter_1", "Identity", "PadV2", "Identity_1", "Merge_1", "Enter_2", "Merge_2", "Switch_1",
    "GatherV2_6", "Pack_3", "C_102", "Enter_3", "NextIteration_1", "Enter_4", "NextIteration_2",
    "Enter_5", "GatherV2_7", "C_103", "Pack_4", "Identity_2", "C_104", "Add_6", "C_105", "Add_7",
    "StridedSlice_23", "NonMaxSuppressionV3", "C_106", "C_107", "Sub_6", "Switch_2", "C_108", "C_109",
    "Where_2", "C_110", "C_111", "C_112", "GatherV2_8", "GatherV2_9", "C_113", "C_114", "C_115",
    "C_116", "StridedSlice_24", "Equal", "Enter_6", "C_117", "Enter_7", "C_118", "Shape", "C_119",
    "C_120", "C_121", "Enter_8", "TensorArrayReadV3", "GatherV2_10", "GatherV2_11", "GatherV2_12",
    "Enter_9", "Enter_10", "C_122", "C_123", "C_124", "TensorArrayV3_1", "TensorArrayScatterV3",
    "C_125", "Unique"],
"src_inter_flow": [["Pack:out0", "Reshape:in0"], ["C:out0", "Reshape:in1"], ["Pad:out0", "Pack:in0"],
    ["ConcatV2:out0", "Pad:in0"], ["C_1:out0", "Pad:in1"], ["GatherV2:out0", "ConcatV2:in0"],
    ["StridedSlice:out0", "ConcatV2:in1"], ["StridedSlice_1:out0", "ConcatV2:in2"],
    ["C_2:out0", "ConcatV2:in3"], ["ConcatV2_1:out0", "GatherV2:in0"],
    ["GatherV2_1:out0", "GatherV2:in1"], ["C_3:out0", "GatherV2:in2"],
    ["Cast:out0", "StridedSlice:in0"], ["C_4:out0", "StridedSlice:in1"],
    ["C_5:out0", "StridedSlice:in2"], ["C_6:out0", "StridedSlice:in3"],
    ["GatherV2_2:out0", "StridedSlice_1:in0"], ["C_7:out0", "StridedSlice_1:in1"],
    ["C_8:out0", "StridedSlice_1:in2"], ["C_9:out0", "StridedSlice_1:in3"],
    ["Maximum:out0", "ConcatV2_1:in0"], ["Maximum_1:out0", "ConcatV2_1:in1"],
    ["Maximum_2:out0", "ConcatV2_1:in2"], ["Maximum_3:out0", "ConcatV2_1:in3"],
    ["C_10:out0", "ConcatV2_1:in4"], ["StridedSlice_2:out0", "GatherV2_1:in0"],
    ["TopKV2:out1", "GatherV2_1:in1"], ["C_11:out0", "GatherV2_1:in2"],
    ["GatherV2_3:out0", "Cast:in0"], ["GatherV2_1:out0", "GatherV2_2:in1"],
    ["GatherNd:out0", "GatherV2_2:in0"], ["C_12:out0", "GatherV2_2:in2"],
    ["Minimum:out0", "Maximum:in0"], ["Split:out0", "Maximum:in1"], ["Split:out1", "Maximum_1:in1"],
    ["Minimum_1:out0", "Maximum_1:in0"], ["Split:out0", "Maximum_2:in1"],
    ["Minimum_2:out0", "Maximum_2:in0"], ["Split:out1", "Maximum_3:in1"],
    ["Minimum_3:out0", "Maximum_3:in0"], ["SparseToDense:out0", "StridedSlice_2:in0"],
    ["C_13:out0", "StridedSlice_2:in1"], ["C_14:out0", "StridedSlice_2:in2"],
    ["C_15:out0", "StridedSlice_2:in3"], ["GatherV2_1:out0", "GatherV2_3:in1"],
    ["GatherV2_4:out0", "TopKV2:in0"], ["C_16:out0", "TopKV2:in1"], ["ArgMax:out0", "GatherV2_3:in0"],
    ["C_17:out0", "GatherV2_3:in2"], ["StridedSlice_3:out0", "GatherNd:in0"],
    ["Pack_1:out0", "GatherNd:in1"], ["Split:out2", "Minimum:in1"], ["Split_1:out0", "Minimum:in0"],
    ["Split:out3", "Minimum_1:in1"], ["C_18:out0", "Split:in0"], ["StridedSlice_4:out0", "Split:in1"],
    ["Split:out2", "Minimum_2:in1"], ["Split_1:out1", "Minimum_1:in0"],
    ["Split:out3", "Minimum_3:in1"], ["Split_1:out2", "Minimum_2:in0"],
    ["Split_1:out3", "Minimum_3:in0"], ["DenseToDenseSetOperation:out1", "SparseToDense:in2"],
    ["DenseToDenseSetOperation:out2", "SparseToDense:in1"],
    ["DenseToDenseSetOperation:out0", "SparseToDense:in0"], ["C_19:out0", "SparseToDense:in3"],
    ["StridedSlice_2:out0", "GatherV2_4:in1"], ["GatherNd:out0", "GatherV2_4:in0"],
    ["C_20:out0", "GatherV2_4:in2"], ["StridedSlice_3:out0", "ArgMax:in0"],
    ["C_21:out0", "ArgMax:in1"], ["ArgMax:out0", "Pack_1:in1"], ["C_22:out0", "StridedSlice_3:in1"],
    ["C_23:out0", "StridedSlice_3:in2"], ["C_24:out0", "StridedSlice_3:in3"],
    ["C_25:out0", "Pack_1:in0"], ["C_26:out0", "Split_1:in0"], ["Pack_2:out0", "Split_1:in1"],
    ["RealDiv:out0", "StridedSlice_4:in0"], ["C_27:out0", "StridedSlice_4:in1"],
    ["C_28:out0", "StridedSlice_4:in2"], ["C_29:out0", "StridedSlice_4:in3"],
    ["ExpandDims:out0", "DenseToDenseSetOperation:in0"],
    ["ExpandDims_1:out0", "DenseToDenseSetOperation:in1"], ["Sub:out0", "Pack_2:in0"],
    ["Sub_1:out0", "Pack_2:in1"], ["Add:out0", "Pack_2:in2"], ["Add_1:out0", "Pack_2:in3"],
    ["Sub_2:out0", "RealDiv:in0"], ["Sub_3:out0", "RealDiv:in1"],
    ["StridedSlice_5:out0", "ExpandDims:in0"], ["C_30:out0", "ExpandDims:in1"],
    ["GatherV2_5:out0", "ExpandDims_1:in0"], ["C_31:out0", "ExpandDims_1:in1"],
    ["Add_2:out0", "Sub:in0"], ["Mul:out0", "Sub:in1"], ["Sub:out0", "Add:in0"],
    ["Add_3:out0", "Sub_1:in0"], ["Mul_1:out0", "Sub_1:in1"], ["Sub_1:out0", "Add_1:in0"],
    ["Mul_2:out0", "Add:in1"], ["Mul_3:out0", "Add_1:in1"], ["StridedSlice_6:out0", "Sub_2:in0"],
    ["C_32:out0", "Sub_2:in1"], ["ConcatV2_2:out0", "Sub_3:in0"], ["C_33:out0", "Sub_3:in1"],
    ["Where:out0", "StridedSlice_5:in0"], ["C_34:out0", "StridedSlice_5:in1"],
    ["C_35:out0", "StridedSlice_5:in2"], ["C_36:out0", "StridedSlice_5:in3"],
    ["Reshape_1:out0", "GatherV2_5:in0"], ["StridedSlice_7:out0", "GatherV2_5:in1"],
    ["C_37:out0", "GatherV2_5:in2"], ["Mul_2:out0", "Mul:in1"], ["Add_4:out0", "Add_2:in0"],
    ["Mul_4:out0", "Add_2:in1"], ["C_38:out0", "Mul:in0"], ["Mul_3:out0", "Mul_1:in1"],
    ["Add_5:out0", "Add_3:in0"], ["Mul_5:out0", "Add_3:in1"], ["C_39:out0", "Mul_1:in0"],
    ["Sub_4:out0", "Mul_2:in0"], ["Exp:out0", "Mul_2:in1"], ["Sub_5:out0", "Mul_3:in0"],
    ["Exp_1:out0", "Mul_3:in1"], ["C_40:out0", "StridedSlice_6:in1"],
    ["C_41:out0", "StridedSlice_6:in2"], ["C_42:out0", "StridedSlice_6:in3"],
    ["Split_2:out0", "ConcatV2_2:in2"], ["Split_2:out1", "ConcatV2_2:in3"],
    ["Split_2:out0", "ConcatV2_2:in0"], ["Split_2:out1", "ConcatV2_2:in1"],
    ["C_43:out0", "ConcatV2_2:in4"], ["Greater:out0", "Where:in0"],
    ["TensorArrayGatherV3:out0", "Reshape_1:in0"], ["C_44:out0", "Reshape_1:in1"],
    ["Where_1:out0", "StridedSlice_7:in0"], ["C_45:out0", "StridedSlice_7:in1"],
    ["C_46:out0", "StridedSlice_7:in2"], ["C_47:out0", "StridedSlice_7:in3"],
    ["Sub_4:out0", "Mul_4:in1"], ["StridedSlice_8:out0", "Add_4:in0"], ["Mul_6:out0", "Add_4:in1"],
    ["StridedSlice_9:out0", "Mul_4:in0"], ["Sub_5:out0", "Mul_5:in1"],
    ["StridedSlice_10:out0", "Add_5:in0"], ["Mul_7:out0", "Add_5:in1"],
    ["StridedSlice_11:out0", "Mul_5:in0"], ["StridedSlice_12:out0", "Sub_4:in0"],
    ["StridedSlice_13:out0", "Sub_4:in1"], ["StridedSlice_14:out0", "Exp:in0"],
    ["ArgMax:out0", "Greater:in0"], ["StridedSlice_15:out0", "Sub_5:in0"],
    ["StridedSlice_16:out0", "Sub_5:in1"], ["StridedSlice_17:out0", "Exp_1:in0"],
    ["C_48:out0", "Split_2:in0"], ["StridedSlice_18:out0", "Split_2:in1"],
    ["C_49:out0", "Greater:in1"], ["TensorArrayV3:out0", "TensorArrayGatherV3:in0"],
    ["Range:out0", "TensorArrayGatherV3:in1"], ["Exit:out0", "TensorArrayGatherV3:in2"],
    ["Greater_1:out0", "Where_1:in0"], ["Sub_4:out0", "Mul_6:in1"],
    ["StridedSlice_19:out0", "StridedSlice_8:in0"], ["C_50:out0", "StridedSlice_8:in1"],
    ["C_51:out0", "StridedSlice_8:in2"], ["C_52:out0", "StridedSlice_8:in3"],
    ["Sub_5:out0", "Mul_7:in1"], ["C_53:out0", "Mul_6:in0"], ["Mul_8:out0", "StridedSlice_9:in0"],
    ["C_54:out0", "StridedSlice_9:in1"], ["C_55:out0", "StridedSlice_9:in2"],
    ["C_56:out0", "StridedSlice_9:in3"], ["StridedSlice_19:out0", "StridedSlice_10:in0"],
    ["C_57:out0", "StridedSlice_10:in1"], ["C_58:out0", "StridedSlice_10:in2"],
    ["C_59:out0", "StridedSlice_10:in3"], ["C_60:out0", "Mul_7:in0"],
    ["Mul_8:out0", "StridedSlice_11:in0"], ["StridedSlice_19:out0", "StridedSlice_12:in0"],
    ["C_61:out0", "StridedSlice_11:in1"], ["C_62:out0", "StridedSlice_11:in2"],
    ["C_63:out0", "StridedSlice_11:in3"], ["StridedSlice_19:out0", "StridedSlice_13:in0"],
    ["C_64:out0", "StridedSlice_12:in1"], ["C_65:out0", "StridedSlice_12:in2"],
    ["C_66:out0", "StridedSlice_12:in3"], ["Mul_8:out0", "StridedSlice_14:in0"],
    ["C_67:out0", "StridedSlice_13:in1"], ["StridedSlice_19:out0", "StridedSlice_15:in0"],
    ["C_68:out0", "StridedSlice_13:in2"], ["C_69:out0", "StridedSlice_13:in3"],
    ["StridedSlice_19:out0", "StridedSlice_16:in0"], ["C_70:out0", "StridedSlice_14:in1"],
    ["C_71:out0", "StridedSlice_14:in2"], ["C_72:out0", "StridedSlice_14:in3"],
    ["C_73:out0", "StridedSlice_15:in1"], ["Mul_8:out0", "StridedSlice_17:in0"],
    ["C_74:out0", "StridedSlice_15:in2"], ["C_75:out0", "StridedSlice_15:in3"],
    ["C_76:out0", "StridedSlice_16:in1"], ["C_77:out0", "StridedSlice_16:in2"],
    ["C_78:out0", "StridedSlice_16:in3"], ["Reshape_1:out0", "Greater_1:in0"],
    ["C_79:out0", "StridedSlice_17:in1"], ["C_80:out0", "StridedSlice_17:in2"],
    ["C_81:out0", "StridedSlice_17:in3"], ["StridedSlice_20:out0", "StridedSlice_18:in0"],
    ["C_82:out0", "StridedSlice_18:in1"], ["C_83:out0", "StridedSlice_18:in2"],
    ["C_84:out0", "StridedSlice_18:in3"], ["C_85:out0", "TensorArrayV3:in0"],
    ["C_86:out0", "Range:in0"], ["TensorArraySizeV3:out0", "Range:in1"], ["C_87:out0", "Range:in2"],
    ["Switch:out0", "Exit:in0"], ["C_88:out0", "Greater_1:in1"], ["C_89:out0", "StridedSlice_19:in1"],
    ["C_90:out0", "StridedSlice_19:in2"], ["C_91:out0", "StridedSlice_19:in3"],
    ["GatherNd_1:out0", "Mul_8:in0"], ["C_92:out0", "Mul_8:in1"], ["Pack_1:out0", "GatherNd_1:in1"],
    ["StridedSlice_21:out0", "StridedSlice_20:in0"], ["C_93:out0", "StridedSlice_20:in1"],
    ["C_94:out0", "StridedSlice_20:in2"], ["C_95:out0", "StridedSlice_20:in3"],
    ["TensorArrayV3:out0", "TensorArraySizeV3:in0"], ["Exit:out0", "TensorArraySizeV3:in1"],
    ["Merge:out0", "Switch:in0"], ["LoopCond:out0", "Switch:in1"],
    ["StridedSlice_22:out0", "GatherNd_1:in0"], ["C_96:out0", "StridedSlice_21:in1"],
    ["C_97:out0", "StridedSlice_21:in2"], ["C_98:out0", "StridedSlice_21:in3"],
    ["TensorArrayV3:out1", "Enter:in0"], ["Enter:out0", "Merge:in0"],
    ["NextIteration:out0", "Merge:in1"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["C_99:out0", "StridedSlice_22:in1"], ["C_100:out0", "StridedSlice_22:in2"],
    ["C_101:out0", "StridedSlice_22:in3"], ["TensorArrayWriteV3:out0", "NextIteration:in0"],
    ["TensorArrayV3:out0", "Enter_1:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["Less_1:out0", "LogicalAnd:in1"], ["Enter_1:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["PadV2:out0", "TensorArrayWriteV3:in2"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["Merge_1:out0", "Less:in0"],
    ["Enter_2:out0", "Less:in1"], ["Enter_2:out0", "Less_1:in1"], ["Merge_2:out0", "Less_1:in0"],
    ["Switch:out1", "Identity_1:in0"], ["Switch_1:out1", "Identity:in0"],
    ["StridedSlice_5:out0", "Enter_5:in0"], ["C_85:out0", "Enter_2:in0"],
    ["GatherV2_6:out0", "PadV2:in0"], ["Pack_3:out0", "PadV2:in1"], ["C_102:out0", "PadV2:in2"],
    ["Enter_3:out0", "Merge_1:in0"], ["NextIteration_1:out0", "Merge_1:in1"],
    ["LoopCond:out0", "Switch_1:in1"], ["Enter_4:out0", "Merge_2:in0"],
    ["NextIteration_2:out0", "Merge_2:in1"], ["Merge_2:out0", "Switch_1:in0"],
    ["Enter_5:out0", "GatherV2_6:in0"], ["GatherV2_7:out0", "GatherV2_6:in1"],
    ["C_103:out0", "GatherV2_6:in2"], ["Pack_4:out0", "Pack_3:in0"],
    ["Identity_2:out4096", "C_102:in0"], ["C_104:out0", "Enter_3:in0"],
    ["Add_6:out0", "NextIteration_1:in0"], ["C_105:out0", "Enter_4:in0"],
    ["Add_7:out0", "NextIteration_2:in0"], ["StridedSlice_23:out0", "GatherV2_7:in0"],
    ["NonMaxSuppressionV3:out0", "GatherV2_7:in1"], ["C_106:out0", "GatherV2_7:in2"],
    ["Identity_2:out4096", "C_103:in0"], ["C_107:out0", "Pack_4:in0"], ["Sub_6:out0", "Pack_4:in1"],
    ["Switch_2:out1", "Identity_2:in0"], ["Identity_2:out0", "Add_6:in0"],
    ["Identity:out0", "Add_7:in0"], ["C_108:out0", "Add_6:in1"], ["C_109:out0", "Add_7:in1"],
    ["Where_2:out0", "StridedSlice_23:in0"], ["C_110:out0", "StridedSlice_23:in1"],
    ["C_111:out0", "StridedSlice_23:in2"], ["C_112:out0", "StridedSlice_23:in3"],
    ["LoopCond:out0", "Switch_2:in1"], ["Identity_2:out4096", "C_106:in0"],
    ["GatherV2_8:out0", "NonMaxSuppressionV3:in0"], ["GatherV2_9:out0", "NonMaxSuppressionV3:in1"],
    ["C_113:out0", "NonMaxSuppressionV3:in2"], ["C_114:out0", "NonMaxSuppressionV3:in3"],
    ["Identity_2:out4096", "C_107:in0"], ["C_115:out0", "NonMaxSuppressionV3:in4"],
    ["Merge_1:out0", "Switch_2:in0"], ["Identity_2:out4096", "C_108:in0"], ["C_116:out0", "Sub_6:in0"],
    ["StridedSlice_24:out0", "Sub_6:in1"], ["Identity_2:out4096", "C_109:in0"],
    ["ConcatV2_1:out0", "GatherV2_10:in0"], ["Identity_2:out4096", "C_110:in0"],
    ["Identity_2:out4096", "C_111:in0"], ["Equal:out0", "Where_2:in0"],
    ["Identity_2:out4096", "C_112:in0"], ["GatherNd:out0", "GatherV2_11:in0"],
    ["StridedSlice_23:out0", "GatherV2_8:in1"], ["StridedSlice_23:out0", "GatherV2_9:in1"],
    ["Identity_2:out4096", "C_113:in0"], ["ArgMax:out0", "GatherV2_12:in0"],
    ["Enter_6:out0", "GatherV2_8:in0"], ["C_117:out0", "GatherV2_8:in2"],
    ["Identity_2:out4096", "C_114:in0"], ["StridedSlice_5:out0", "GatherV2_10:in1"],
    ["Enter_7:out0", "GatherV2_9:in0"], ["C_118:out0", "GatherV2_9:in2"],
    ["Identity_2:out4096", "C_115:in0"], ["StridedSlice_5:out0", "GatherV2_11:in1"],
    ["Identity_2:out4096", "C_116:in0"], ["StridedSlice_5:out0", "GatherV2_12:in1"],
    ["Shape:out0", "StridedSlice_24:in0"], ["Identity_2:out4096", "C_117:in0"],
    ["C_119:out0", "StridedSlice_24:in1"], ["C_120:out0", "StridedSlice_24:in2"],
    ["C_121:out0", "StridedSlice_24:in3"], ["Enter_8:out0", "Equal:in0"],
    ["TensorArrayReadV3:out0", "Equal:in1"], ["GatherV2_6:out0", "Shape:in0"],
    ["Identity_2:out4096", "C_118:in0"], ["GatherV2_10:out0", "Enter_6:in0"],
    ["Identity_2:out4096", "C_119:in0"], ["GatherV2_11:out0", "Enter_7:in0"],
    ["Identity_2:out4096", "C_120:in0"], ["Identity_2:out4096", "C_121:in0"],
    ["Identity:out0", "TensorArrayReadV3:in1"], ["GatherV2_12:out0", "Enter_8:in0"],
    ["Enter_9:out0", "TensorArrayReadV3:in0"], ["Enter_10:out0", "TensorArrayReadV3:in2"],
    ["C_85:out0", "TensorArrayV3_1:in0"], ["C_122:out0", "GatherV2_10:in2"],
    ["C_123:out0", "GatherV2_11:in2"], ["C_124:out0", "GatherV2_12:in2"],
    ["TensorArrayV3_1:out0", "Enter_9:in0"], ["TensorArrayScatterV3:out0", "Enter_10:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"],
    ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"], ["C_125:out0", "TensorArrayScatterV3:in1"],
    ["Unique:out0", "TensorArrayScatterV3:in2"], ["GatherV2_12:out0", "Unique:in0"]],
"src_in_anchor": [["I:out0", "StridedSlice_21:in0"], ["I:out0", "StridedSlice_6:in0"],
    ["I_1:out0", "StridedSlice_19:in0"], ["I_2:out0", "StridedSlice_3:in0"],
    ["I_3:out0", "StridedSlice_22:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["mrcnn_detection"],
"src_acu_in_tensor_map": [["I:out0", "mrcnn_detection:in0"], ["I_1:out0", "mrcnn_detection:in1"],
                          ["I_2:out0", "mrcnn_detection:in2"], ["I_3:out0", "mrcnn_detection:in3"]],
"src_acu_out_tensor_map": [["Reshape:out0", "mrcnn_detection:out0"]],
"acu_inter_flow": [],
"param_map": {"mrcnn_detection": {}},
"blob_map": {"mrcnn_detection": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mrcnn_detection)

r_mrcnn_roi_align = {
"ruler_name": "mrcnn-roi_align",
"src_ops_alias": ["Reshape", "GatherV2", "C", "ConcatV2", "GatherV2_1", "C_1", "CropAndResize", "CropAndResize_1",
    "CropAndResize_2", "CropAndResize_3", "C_2", "StridedSlice", "StridedSlice_1", "C_3",
    "StopGradient", "StopGradient_1", "C_4", "StopGradient_2", "StopGradient_3", "C_5",
    "StopGradient_4", "StopGradient_5", "C_6", "StopGradient_6", "StopGradient_7", "C_7", "ConcatV2_1",
    "C_8", "C_9", "C_10", "TopKV2", "C_11", "C_12", "C_13", "GatherNd", "Cast", "GatherNd_1", "Cast_1",
    "GatherNd_2", "Cast_2", "GatherNd_3", "Cast_3", "Cast_4", "C_14", "C_15", "Add", "C_16", "Where",
    "StridedSlice_2", "Where_1", "StridedSlice_3", "Where_2", "StridedSlice_4", "Where_3",
    "StridedSlice_5", "ConcatV2_2", "Mul", "StridedSlice_6", "Equal", "C_17", "C_18", "C_19",
    "Equal_1", "C_20", "C_21", "C_22", "Equal_2", "C_23", "C_24", "C_25", "Equal_3", "C_26", "C_27",
    "C_28", "C_29", "StridedSlice_7", "C_30", "C_31", "C_32", "C_33", "Squeeze", "C_34", "C_35",
    "C_36", "C_37", "C_38", "C_39", "C_40", "Minimum", "C_41", "Maximum", "C_42", "Add_1", "C_43",
    "Cast_5", "Round", "RealDiv", "Log", "C_44", "RealDiv_1", "Sqrt", "RealDiv_2", "Mul_1", "C_45",
    "Sqrt_1", "Sub", "Sub_1", "Mul_2", "Split", "StridedSlice_8", "StridedSlice_9", "C_46",
    "StridedSlice_10", "C_47", "C_48", "C_49", "C_50", "C_51", "C_52", "StridedSlice_11", "C_53",
    "C_54", "C_55", "C_56", "C_57", "C_58"],
"src_inter_flow": [["GatherV2:out0", "Reshape:in0"], ["C:out0", "Reshape:in1"], ["ConcatV2:out0", "GatherV2:in0"],
    ["GatherV2_1:out0", "GatherV2:in1"], ["C_1:out0", "GatherV2:in2"],
    ["CropAndResize:out0", "ConcatV2:in0"], ["CropAndResize_1:out0", "ConcatV2:in1"],
    ["CropAndResize_2:out0", "ConcatV2:in2"], ["CropAndResize_3:out0", "ConcatV2:in3"],
    ["C_2:out0", "ConcatV2:in4"], ["StridedSlice:out0", "GatherV2_1:in0"],
    ["StridedSlice_1:out0", "GatherV2_1:in1"], ["C_3:out0", "GatherV2_1:in2"],
    ["StopGradient:out0", "CropAndResize:in1"], ["StopGradient_1:out0", "CropAndResize:in2"],
    ["C_4:out0", "CropAndResize:in3"], ["StopGradient_2:out0", "CropAndResize_1:in1"],
    ["StopGradient_3:out0", "CropAndResize_1:in2"], ["C_5:out0", "CropAndResize_1:in3"],
    ["StopGradient_4:out0", "CropAndResize_2:in1"], ["StopGradient_5:out0", "CropAndResize_2:in2"],
    ["C_6:out0", "CropAndResize_2:in3"], ["StopGradient_6:out0", "CropAndResize_3:in1"],
    ["StopGradient_7:out0", "CropAndResize_3:in2"], ["C_7:out0", "CropAndResize_3:in3"],
    ["ConcatV2_1:out0", "StridedSlice:in0"], ["C_8:out0", "StridedSlice:in1"],
    ["C_9:out0", "StridedSlice:in2"], ["C_10:out0", "StridedSlice:in3"],
    ["TopKV2:out1", "StridedSlice_1:in0"], ["C_11:out0", "StridedSlice_1:in1"],
    ["C_12:out0", "StridedSlice_1:in2"], ["C_13:out0", "StridedSlice_1:in3"],
    ["GatherNd:out0", "StopGradient:in0"], ["Cast:out0", "StopGradient_1:in0"],
    ["GatherNd_1:out0", "StopGradient_2:in0"], ["Cast_1:out0", "StopGradient_3:in0"],
    ["GatherNd_2:out0", "StopGradient_4:in0"], ["Cast_2:out0", "StopGradient_5:in0"],
    ["GatherNd_3:out0", "StopGradient_6:in0"], ["Cast_3:out0", "StopGradient_7:in0"],
    ["Cast_4:out0", "ConcatV2_1:in0"], ["C_14:out0", "ConcatV2_1:in1"],
    ["C_15:out0", "ConcatV2_1:in2"], ["Add:out0", "TopKV2:in0"], ["C_16:out0", "TopKV2:in1"],
    ["Where:out0", "GatherNd:in1"], ["StridedSlice_2:out0", "Cast:in0"],
    ["Where_1:out0", "GatherNd_1:in1"], ["StridedSlice_3:out0", "Cast_1:in0"],
    ["Where_2:out0", "GatherNd_2:in1"], ["StridedSlice_4:out0", "Cast_2:in0"],
    ["Where_3:out0", "GatherNd_3:in1"], ["StridedSlice_5:out0", "Cast_3:in0"],
    ["ConcatV2_2:out0", "Cast_4:in0"], ["Mul:out0", "Add:in0"], ["StridedSlice_6:out0", "Add:in1"],
    ["Where:out0", "StridedSlice_2:in0"], ["Equal:out0", "Where:in0"],
    ["C_17:out0", "StridedSlice_2:in1"], ["C_18:out0", "StridedSlice_2:in2"],
    ["C_19:out0", "StridedSlice_2:in3"], ["Where_1:out0", "StridedSlice_3:in0"],
    ["Equal_1:out0", "Where_1:in0"], ["C_20:out0", "StridedSlice_3:in1"],
    ["C_21:out0", "StridedSlice_3:in2"], ["C_22:out0", "StridedSlice_3:in3"],
    ["Where_2:out0", "StridedSlice_4:in0"], ["Equal_2:out0", "Where_2:in0"],
    ["C_23:out0", "StridedSlice_4:in1"], ["C_24:out0", "StridedSlice_4:in2"],
    ["C_25:out0", "StridedSlice_4:in3"], ["Where_3:out0", "StridedSlice_5:in0"],
    ["ConcatV2_1:out0", "StridedSlice_6:in0"], ["Where:out0", "ConcatV2_2:in0"],
    ["Where_1:out0", "ConcatV2_2:in1"], ["Equal_3:out0", "Where_3:in0"],
    ["Where_2:out0", "ConcatV2_2:in2"], ["Where_3:out0", "ConcatV2_2:in3"],
    ["C_26:out0", "StridedSlice_5:in1"], ["C_27:out0", "StridedSlice_5:in2"],
    ["C_28:out0", "StridedSlice_5:in3"], ["C_29:out0", "ConcatV2_2:in4"],
    ["StridedSlice_7:out0", "Mul:in0"], ["C_30:out0", "Mul:in1"], ["C_31:out0", "StridedSlice_6:in1"],
    ["C_32:out0", "StridedSlice_6:in2"], ["C_33:out0", "StridedSlice_6:in3"],
    ["Squeeze:out0", "Equal:in0"], ["C_34:out0", "Equal:in1"], ["Squeeze:out0", "Equal_1:in0"],
    ["C_35:out0", "Equal_1:in1"], ["Squeeze:out0", "Equal_2:in0"], ["C_36:out0", "Equal_2:in1"],
    ["ConcatV2_1:out0", "StridedSlice_7:in0"], ["Squeeze:out0", "Equal_3:in0"],
    ["C_37:out0", "Equal_3:in1"], ["C_38:out0", "StridedSlice_7:in1"],
    ["C_39:out0", "StridedSlice_7:in2"], ["C_40:out0", "StridedSlice_7:in3"],
    ["Minimum:out0", "Squeeze:in0"], ["C_41:out0", "Minimum:in0"], ["Maximum:out0", "Minimum:in1"],
    ["C_42:out0", "Maximum:in0"], ["Add_1:out0", "Maximum:in1"], ["C_43:out0", "Add_1:in0"],
    ["Cast_5:out0", "Add_1:in1"], ["Round:out0", "Cast_5:in0"], ["RealDiv:out0", "Round:in0"],
    ["Log:out0", "RealDiv:in0"], ["C_44:out0", "RealDiv:in1"], ["RealDiv_1:out0", "Log:in0"],
    ["Sqrt:out0", "RealDiv_1:in0"], ["RealDiv_2:out0", "RealDiv_1:in1"], ["Mul_1:out0", "Sqrt:in0"],
    ["C_45:out0", "RealDiv_2:in0"], ["Sqrt_1:out0", "RealDiv_2:in1"], ["Sub:out0", "Mul_1:in0"],
    ["Sub_1:out0", "Mul_1:in1"], ["Mul_2:out0", "Sqrt_1:in0"], ["Split:out0", "Sub:in1"],
    ["Split:out2", "Sub:in0"], ["Split:out3", "Sub_1:in0"], ["Split:out1", "Sub_1:in1"],
    ["StridedSlice_8:out0", "Mul_2:in0"], ["StridedSlice_9:out0", "Mul_2:in1"],
    ["C_46:out0", "Split:in0"], ["StridedSlice_10:out0", "StridedSlice_8:in0"],
    ["C_47:out0", "StridedSlice_8:in1"], ["C_48:out0", "StridedSlice_8:in2"],
    ["C_49:out0", "StridedSlice_8:in3"], ["StridedSlice_10:out0", "StridedSlice_9:in0"],
    ["C_50:out0", "StridedSlice_9:in1"], ["C_51:out0", "StridedSlice_9:in2"],
    ["C_52:out0", "StridedSlice_9:in3"], ["StridedSlice_11:out0", "StridedSlice_10:in0"],
    ["C_53:out0", "StridedSlice_10:in1"], ["C_54:out0", "StridedSlice_10:in2"],
    ["C_55:out0", "StridedSlice_10:in3"], ["C_56:out0", "StridedSlice_11:in1"],
    ["C_57:out0", "StridedSlice_11:in2"], ["C_58:out0", "StridedSlice_11:in3"]],
"src_in_anchor": [["I:out0", "Split:in1"], ["I:out0", "GatherNd:in0"], ["I:out0", "GatherNd_1:in0"],
    ["I:out0", "GatherNd_2:in0"], ["I:out0", "GatherNd_3:in0"], ["I_1:out0", "StridedSlice_11:in0"],
    ["I_2:out0", "CropAndResize:in0"], ["I_3:out0", "CropAndResize_1:in0"],
    ["I_4:out0", "CropAndResize_2:in0"], ["I_5:out0", "CropAndResize_3:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["mrcnn_roi_align"],
"src_acu_in_tensor_map": [["I:out0", "mrcnn_roi_align:in0"], ["I_1:out0", "mrcnn_roi_align:in1"],
                          ["I_2:out0", "mrcnn_roi_align:in2"], ["I_3:out0", "mrcnn_roi_align:in3"],
                          ["I_4:out0", "mrcnn_roi_align:in4"], ["I_5:out0", "mrcnn_roi_align:in5"]],
"src_acu_out_tensor_map": [["Reshape:out0", "mrcnn_roi_align:out0"]],
"acu_inter_flow": [],
"param_map": {"mrcnn_roi_align": {'pool_size': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_4:out0'])"]},},
"blob_map": {"mrcnn_roi_align": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_mrcnn_roi_align)

r_crop_and_resize = {
"ruler_name": "crop_and_resize",
"src_ops_alias": ["CropAndResize", "C", "C_1", "C_2"],
"src_inter_flow": [["C:out0", "CropAndResize:in1"], ["C_1:out0", "CropAndResize:in2"],
                   ["C_2:out0", "CropAndResize:in3"]],
"src_in_anchor": [["I:out0", "CropAndResize:in0"]],
"src_out_tensor": ["CropAndResize:out0"],
"acu_lys_alias": ["cropandresize"],
"src_acu_in_tensor_map": [["I:out0", "cropandresize:in0"]],
"src_acu_out_tensor_map": [["CropAndResize:out0", "cropandresize:out0"]],
"acu_inter_flow": [],
"param_map": {"cropandresize": {'num_crop_boxes': ['INT', 'CODE', "self.shape_pick(tensor['C:out0'])[0]"],
                                'crop_size': ['INTS', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])"],
                                'resize_method': ['STRING', 'CODE',
                                                  "self.attr_pick(node['CropAndResize'], 'method', 'bilinear')"],}},
"blob_map": {"cropandresize": {'boxes': ['CODE', "self.tensor_to_numpy(tensor['C:out0'])"],
                               'box_ind': ['CODE', "self.tensor_to_numpy(tensor['C_1:out0'])"],}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_crop_and_resize)

r_time_major_lstm_2_0_0 = {
"ruler_name": "lstm_broadcom",
"src_ops_alias": ["TensorArrayGatherV3", "TensorArrayV3", "Range", "Exit", "C", "C_1", "TensorArraySizeV3", "C_2",
    "Switch", "Merge", "LoopCond", "Enter", "NextIteration", "LogicalAnd", "TensorArrayWriteV3",
    "Less", "Less_1", "Enter_1", "Identity", "Mul", "Identity_1", "Merge_1", "Enter_2", "Merge_2",
    "Enter_3", "Switch_1", "Sigmoid", "Tanh", "Enter_4", "NextIteration_1", "Enter_5",
    "NextIteration_2", "C_3", "Split", "AddV2", "C_4", "AddV2_1", "C_5", "AddV2_2", "C_6", "BiasAdd",
    "Mul_1", "Mul_2", "Identity_2", "C_7", "C_8", "MatMul", "Enter_6", "Sigmoid_1", "Identity_3",
    "Sigmoid_2", "Tanh_1", "Switch_2", "ConcatV2", "Enter_7", "C_9", "AddV2_3", "Switch_3",
    "TensorArrayReadV3", "Identity_4", "C_10", "C_11", "C_12", "Merge_3", "Enter_8", "Enter_9",
    "Switch_4", "Enter_10", "NextIteration_3", "TensorArrayV3_1", "TensorArrayScatterV3", "Merge_4",
    "C_13", "C_14", "Enter_11", "NextIteration_4", "C_15"],
"src_inter_flow": [["TensorArrayV3:out0", "TensorArrayGatherV3:in0"], ["Range:out0", "TensorArrayGatherV3:in1"],
    ["Exit:out0", "TensorArrayGatherV3:in2"], ["C:out0", "TensorArrayV3:in0"],
    ["C_1:out0", "Range:in0"], ["TensorArraySizeV3:out0", "Range:in1"], ["C_2:out0", "Range:in2"],
    ["Switch:out0", "Exit:in0"], ["TensorArrayV3:out0", "TensorArraySizeV3:in0"],
    ["Exit:out0", "TensorArraySizeV3:in1"], ["Merge:out0", "Switch:in0"],
    ["LoopCond:out0", "Switch:in1"], ["Enter:out0", "Merge:in0"], ["NextIteration:out0", "Merge:in1"],
    ["TensorArrayV3:out1", "Enter:in0"], ["LogicalAnd:out0", "LoopCond:in0"],
    ["TensorArrayWriteV3:out0", "NextIteration:in0"], ["Less:out0", "LogicalAnd:in0"],
    ["Less_1:out0", "LogicalAnd:in1"], ["Enter_1:out0", "TensorArrayWriteV3:in0"],
    ["Identity:out0", "TensorArrayWriteV3:in1"], ["Mul:out0", "TensorArrayWriteV3:in2"],
    ["Identity_1:out0", "TensorArrayWriteV3:in3"], ["Merge_1:out0", "Less:in0"],
    ["TensorArrayV3:out0", "Enter_1:in0"], ["Enter_2:out0", "Less:in1"],
    ["Merge_2:out0", "Less_1:in0"], ["Enter_3:out0", "Less_1:in1"], ["Switch_1:out1", "Identity:in0"],
    ["Switch:out1", "Identity_1:in0"], ["Sigmoid:out0", "Mul:in0"], ["Tanh:out0", "Mul:in1"],
    ["C:out0", "Enter_2:in0"], ["Enter_4:out0", "Merge_1:in0"],
    ["NextIteration_1:out0", "Merge_1:in1"], ["LoopCond:out0", "Switch_1:in1"],
    ["Enter_5:out0", "Merge_2:in0"], ["NextIteration_2:out0", "Merge_2:in1"],
    ["C_3:out0", "Enter_3:in0"], ["Merge_2:out0", "Switch_1:in0"], ["Split:out3", "Sigmoid:in0"],
    ["AddV2:out0", "Tanh:in0"], ["C_4:out0", "Enter_4:in0"], ["AddV2_1:out0", "NextIteration_1:in0"],
    ["C_5:out0", "Enter_5:in0"], ["AddV2_2:out0", "NextIteration_2:in0"], ["C_6:out0", "Split:in0"],
    ["BiasAdd:out0", "Split:in1"], ["Mul_1:out0", "AddV2:in0"], ["Mul_2:out0", "AddV2:in1"],
    ["Identity:out0", "AddV2_2:in0"], ["Identity_2:out0", "AddV2_1:in0"], ["C_7:out0", "AddV2_1:in1"],
    ["C_8:out0", "AddV2_2:in1"], ["Identity_2:out4096", "C_6:in0"], ["MatMul:out0", "BiasAdd:in0"],
    ["Enter_6:out0", "BiasAdd:in1"], ["Sigmoid_1:out0", "Mul_1:in0"], ["Identity_3:out0", "Mul_1:in1"],
    ["Sigmoid_2:out0", "Mul_2:in0"], ["Tanh_1:out0", "Mul_2:in1"], ["Switch_2:out1", "Identity_2:in0"],
    ["Identity_2:out4096", "C_7:in0"], ["Identity_2:out4096", "C_8:in0"],
    ["ConcatV2:out0", "MatMul:in0"], ["Enter_7:out0", "MatMul:in1"], ["C_9:out0", "Enter_6:in0"],
    ["LoopCond:out0", "Switch_2:in1"], ["Split:out0", "Sigmoid_2:in0"],
    ["AddV2_3:out0", "Sigmoid_1:in0"], ["Merge_1:out0", "Switch_2:in0"], ["Split:out1", "Tanh_1:in0"],
    ["Switch_3:out1", "Identity_3:in0"], ["LoopCond:out0", "Switch_3:in1"],
    ["TensorArrayReadV3:out0", "ConcatV2:in0"], ["Identity_4:out0", "ConcatV2:in1"],
    ["C_10:out0", "ConcatV2:in2"], ["Split:out2", "AddV2_3:in0"],
    ["Identity:out0", "TensorArrayReadV3:in1"], ["C_11:out0", "Enter_7:in0"],
    ["C_12:out0", "AddV2_3:in1"], ["Merge_3:out0", "Switch_3:in0"], ["Identity_2:out4096", "C_10:in0"],
    ["Enter_8:out0", "TensorArrayReadV3:in0"], ["Enter_9:out0", "TensorArrayReadV3:in2"],
    ["Switch_4:out1", "Identity_4:in0"], ["Identity_2:out4096", "C_12:in0"],
    ["LoopCond:out0", "Switch_4:in1"], ["C:out0", "TensorArrayV3_1:in0"],
    ["Enter_10:out0", "Merge_3:in0"], ["NextIteration_3:out0", "Merge_3:in1"],
    ["TensorArrayV3_1:out0", "Enter_8:in0"], ["AddV2:out0", "NextIteration_3:in0"],
    ["TensorArrayScatterV3:out0", "Enter_9:in0"], ["Merge_4:out0", "Switch_4:in0"],
    ["C_13:out0", "Enter_10:in0"], ["TensorArrayV3_1:out0", "TensorArrayScatterV3:in0"],
    ["TensorArrayV3_1:out1", "TensorArrayScatterV3:in3"], ["Mul:out0", "NextIteration_4:in0"],
    ["C_14:out0", "TensorArrayScatterV3:in1"], ["Enter_11:out0", "Merge_4:in0"],
    ["NextIteration_4:out0", "Merge_4:in1"], ["C_15:out0", "Enter_11:in0"]],
"src_in_anchor": [["I:out0", "TensorArrayScatterV3:in2"]],
"src_out_tensor": ["TensorArrayGatherV3:out0"],
"acu_lys_alias": ["lstm"],
"src_acu_in_tensor_map": [["I:out0", "lstm:in0"]],
"src_acu_out_tensor_map": [["TensorArrayGatherV3:out0", "lstm:out0"]],
"acu_inter_flow": [],
"param_map": {"lstm": {
    'time_major': ['BOOL', 'VALUE', True],
    'forget_bias': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_12:out0'])"],
    'weights': ['INT', 'CODE', "self.shape_pick(tensor['Enter_7:out0'])[1] / 4"],
    }},
"blob_map": {
    "lstm": {
        'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel': ['CODE', "self.tensor_to_numpy(tensor['C_11:out0'])"],
        'rnn/multi_rnn_cell/cell_0/lstm_cell/bias': ['CODE', "self.tensor_to_numpy(tensor['C_9:out0'])"],
    }},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_time_major_lstm_2_0_0)

r_round = {
    "ruler_name": "round",
    "src_ops_alias": ["Round"],
    "src_inter_flow": [],
    "src_in_anchor": [["I:out0", "Round:in0"]],
    "src_out_tensor": ["Round:out0"],
    "acu_lys_alias": ["round"],
    "src_acu_in_tensor_map": [["I:out0", "round:in0"]],
    "src_acu_out_tensor_map": [["Round:out0", "round:out0"]],
    "acu_inter_flow": [],
    "param_map": {"round": {}},
    "blob_map": {"round": {}},
    "priority_tip": 0,
    "pre_condition": None}
ruler_list.append(r_round)

r_sequencemask_with_1input = {
"ruler_name": "sequencemask_with_maxlen",
"src_ops_alias": ["Less", "C", "Cast", "ExpandDims", "C_1"],
"src_inter_flow": [["C:out0", "Less:in0"], ["Cast:out0", "Less:in1"], ["ExpandDims:out0", "Cast:in0"],
    ["C_1:out0", "ExpandDims:in1"]],
"src_in_anchor": [["I:out0", "ExpandDims:in0"]],
"src_out_tensor": ["Less:out0"],
"acu_lys_alias": ["sequence_mask"],
"src_acu_in_tensor_map": [["I:out0", "sequence_mask:in0"]],
"src_acu_out_tensor_map": [["Less:out0", "sequence_mask:out0"]],
"acu_inter_flow": [],
"param_map": {"sequence_mask": {
    'maxlen': ['INT', 'CODE', "len(list(self.tensor_to_numpy(tensor['C:out0'])))"],
}},
"blob_map": {"sequence_mask": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_sequencemask_with_1input)

r_erf = {
"ruler_name": "r_erf",
"src_ops_alias": ["Erf"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Erf:in0"]],
"src_out_tensor": ["Erf:out0"],
"acu_lys_alias": ["erf"],
"src_acu_in_tensor_map": [["I:out0", "erf:in0"]],
"src_acu_out_tensor_map": [["Erf:out0", "erf:out0"]],
"acu_inter_flow": [],
"param_map": {"erf": {}},
"blob_map": {"erf": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_erf)
#Erf:bert/encoder/layer_10/intermediate/dense/Erf

r_one_hot = {
"ruler_name": "r_one_hot",
"src_ops_alias": ["OneHot", "C", "C_1", "C_2"],
"src_inter_flow": [["C:out0", "OneHot:in1"], ["C_1:out0", "OneHot:in2"], ["C_2:out0", "OneHot:in3"]],
"src_in_anchor": [["I:out0", "OneHot:in0"]],
"src_out_tensor": ["OneHot:out0"],
"acu_lys_alias": ["one_hot"],
"src_acu_in_tensor_map": [["I:out0", "one_hot:in0"]],
"src_acu_out_tensor_map": [["OneHot:out0", "one_hot:out0"]],
"acu_inter_flow": [],
"param_map": {"one_hot": {
    'depth': ['INT', 'CODE', "self.tensor_to_numpy(tensor['C:out0'])[0]"],
    # 1 is for float32
    'dtype': ['ORIGIN', 'CODE', "self.tf_type_enum_to_ac_type(self.attr_pick(node['OneHot'], 'T', 1))"],
    'on_value': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_1:out0'])[0]"],
    'off_value': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['C_2:out0'])[0]"],
    'axis': ['INT', 'CODE', "self.attr_pick(node['OneHot'], 'axis', -1)"],
}},
"blob_map": {"one_hot": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_one_hot)
#OneHot:bert/embeddings/one_hot;
# C:bert/embeddings/one_hot/depth;
# C_1:bert/embeddings/one_hot/on_value;
# C_2:bert/embeddings/one_hot/off_value

def check_duplicated_rulers(rulerlist):
    '''
    check dulicated ruler names, raise error if duplicated rulers found
    '''
    ruler_dict = {}
    for rule in rulerlist:
        if rule['ruler_name'] in ruler_dict:
            ruler_dict[rule['ruler_name']] += 1
        else:
            ruler_dict[rule['ruler_name']] = 1

    duplicated_rulers = [(r, ruler_dict[r]) for r in ruler_dict if ruler_dict[r] > 1]
    if len(duplicated_rulers) > 0:
        content = '\n'.join(['{:>20}: {}'.format(r[0], r[1]) for r in duplicated_rulers])
        raise ValueError('Found duplicated rulers:\n' + content)

def gen_tf_ruler(dst_path):
    # print(json.dumps(ruler_list))
    dst_path = os.path.join(dst_path,'tf_ruler_db.json')

    # check dulicated ruler names, raise error if duplicated rulers found
    check_duplicated_rulers(ruler_list)

    with open(dst_path, 'w+') as f:
        json.dump(ruler_list, f, indent=1)

    # To Verify ruler follow synatx
    with open(dst_path, 'r') as f:
        x_val = json.load(f)

    # For Debug purpose
    # from acuitylib.converter.paragraph import paragraph_serialize
    # db_list = list()
    # for x in x_val:
    #     db_list.append(paragraph_serialize(x))

def main():
    gen_tf_ruler(sys.argv[1])

if  __name__ == '__main__':
    main()
