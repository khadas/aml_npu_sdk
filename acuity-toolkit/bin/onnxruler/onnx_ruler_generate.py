import json
import sys
import os
import dill
import math
#support build-in functions:
'''
    def build_port(self, ly_str, pt=0):
    def have_const_in_inputs(self, tensor):
    def have_single_tensor_in_inputs(self, tensor):
    def input_port_is_const(self, tensor, pt):
    def shape_pick(self, tensor):
    def attr_pick(self, tensor, key, default=0):
    def array_layout(self, array, layout):
    def tensor_to_numpy(self, tensor_name, trans=None):
    def squeeze_shapes(self, squeeze_dims, input_shape):
    def fc_weight(self, in_tensor, node):
    def split_slice_cale(self, slice):
    def map_pad_value(self, node):
    def reducex_axis_list(self, node, input_shape)
'''

ruler_list = list()

def rule_pyfunc_def(func):
    def _wrap_func(*args, **kwargs):
        src = dill.source.getsource(func)
        src = src.replace('\r\n', '\n')
        src = src.replace('@rule_pyfunc_def\n', '')
        src = src.split('\n')
        return ['__rule_func_additional_args = ' + json.dumps(kwargs)] + src if len(kwargs) > 0 else src
    return _wrap_func

@rule_pyfunc_def
def r_softmax_get_sf_axis(self, node, tensor):
    axis = self.attr_pick(node['Softmax'], 'axis', None)
    if axis is None:
        shape = self.shape_pick(tensor['I:out0'])
        if len(shape) == 4 and self.minor_version < 13:
            axis = 1
        else:
            axis = -1
    return axis

@rule_pyfunc_def
def r_softmax_get_log_sf_axis(self, node, tensor):
    axis = self.attr_pick(node['LogSoftmax'], 'axis', None)
    if axis is None:
        shape = self.shape_pick(tensor['I:out0'])
        if len(shape) == 4 and self.minor_version < 13:
            axis = 1
        else:
            axis = -1
    return axis

@rule_pyfunc_def
def r_softmax_pre_cond(self, node, tensor):
    axis = self.attr_pick(node['Softmax'], 'axis', -1)
    input_rank = len(self.shape_pick(tensor['I:out0']))
    if axis < 0:
        axis = input_rank + axis
    if self.minor_version < 13:
        if input_rank > 2:
            if axis == input_rank - 1:
                return True
        elif input_rank == 2:
            return True
        return False
    return True


@rule_pyfunc_def
def r_logsoftmax_pre_cond(self, node, tensor):
    axis = self.attr_pick(node['LogSoftmax'], 'axis', -1)
    input_rank = len(self.shape_pick(tensor['I:out0']))
    if axis < 0:
        axis = input_rank + axis
    if self.minor_version < 13:
        if input_rank > 2:
            if axis == input_rank - 1:
                return True
        elif input_rank == 2:
            return True
        return False
    return True

@rule_pyfunc_def
def r_slice_get_size(self, node, tensor):
    starts = self.attr_pick(node['Slice'], 'starts', None)
    ends = self.attr_pick(node['Slice'], 'ends', None)
    axes = self.attr_pick(node['Slice'], 'axes', None)
    in_shape = self.shape_pick(tensor['I:out0'])
    out_shape = self.shape_pick(tensor['Slice:out0'])

    import numpy as np
    import copy
    INT_MAX = np.iinfo(np.int64).max
    in_shape = copy.deepcopy(in_shape)
    ends = copy.deepcopy(ends)
    size = copy.deepcopy(in_shape)
    for i in range(len(axes)):
        if starts[i] < 0:
            starts[i] = in_shape[axes[i]] + starts[i]
        if ends[i] == INT_MAX:
            ends[i] = in_shape[axes[i]]
        elif ends[i] < 0:
            ends[i] = in_shape[axes[i]] + ends[i]
        size[axes[i]] = ends[i] - starts[i]
    return size

@rule_pyfunc_def
def r_slice_get_begin(self, node, tensor):
    in_shape = self.shape_pick(tensor['I:out0'])
    starts = self.attr_pick(node['Slice'], 'starts', None)
    axes = self.attr_pick(node['Slice'], 'axes', None)
    begin = [0] * len(in_shape)
    for i in range(len(axes)):
        if starts[i] < 0:
            starts[i] = in_shape[axes[i]] + starts[i]
        begin[axes[i]] = starts[i]
    return begin

@rule_pyfunc_def
def r_slice_pre_cond(self, node, tensor, steps_tensor):
    steps = self.tensor_to_numpy(tensor[steps_tensor]).tolist()
    for step in steps:
        if step != 1:
            return False
    return True

@rule_pyfunc_def
def r_get_deconv_weights(self, node, tensor, weight):
    in_channel = self.shape_pick(tensor[weight])[1]
    group = self.attr_pick(node['ConvTranspose'], 'group', 1)
    weights = in_channel * group
    return weights

@rule_pyfunc_def
def r_group_conv1d_pre_condition(self, node, tensor):
    ret = False
    if len(self.shape_pick(tensor['Constant_0:out0'])) == 3:
        in_shape = self.shape_pick(tensor['I:out0'])
        group_number = self.attr_pick(node['Conv'], 'group', 1)
        if group_number > 1:
            ret = True
    return ret


@rule_pyfunc_def
def r_depthwise_conv1d_pre_condition(self, node, tensor):
    ret = False
    if len(self.shape_pick(tensor['Constant_0:out0'])) == 3:
        in_shape = self.shape_pick(tensor['I:out0'])
        group_number = self.attr_pick(node['Conv'], 'group', 1)
        if group_number > 1 and group_number == in_shape[1]:
            ret = True
    return ret

@rule_pyfunc_def
def r_pad_value_map(self, node, tensor):
    pad_np = self.tensor_to_numpy(tensor['Constant:out0'])
    pads = list(pad_np)
    pads = [int(p) for p in pads]
    dims = len(pads) // 2
    pads_array = list()
    if dims == 4:
        for id in range(dims):
            pad = [pads[id], pads[dims + id]]
            pads_array.append(pad)
    elif dims < 4:
        for id in range(dims):
            pad = [pads[id], pads[dims + id]]
            pads_array.append(pad)
    return pads_array

@rule_pyfunc_def
def r_dconv_get_kernel_shape(self, node, tensor, dim):
    kernel_shape = self.attr_pick(node['ConvTranspose'], 'kernel_shape')
    if not kernel_shape:
        kernel = self.tensor_to_numpy(tensor['Constant_0:out0'])
        kernel_shape = kernel.shape
        ksize_h = kernel_shape[2]
        ksize_w = kernel_shape[3]
    else:
        ksize_h = kernel_shape[0]
        ksize_w = kernel_shape[1]
    if dim == 'height':
        return ksize_h
    if dim == 'width':
        return ksize_w

@rule_pyfunc_def
def r_conv1d_get_kernel_shape(self, node, tensor, kernel_name):
    kernel_shape = self.attr_pick(node['Conv'], 'kernel_shape', None)
    if kernel_shape is None:
        kernel = self.tensor_to_numpy(tensor[kernel_name])
        kernel_shape = kernel.shape
        return kernel_shape[2]

    return kernel_shape[0]

@rule_pyfunc_def
def r_permute_value(self, node, tensor):
    in_shape = self.shape_pick(tensor['I:out0'])
    perm = self.attr_pick(node['Transpose'], 'perm', None)
    if perm is None:
        perm = list()
        for idx in range(len(in_shape)):
            perm.append(idx)
        perm.reverse()
    _perm = " ".join([str(x) for x in perm])
    return _perm

@rule_pyfunc_def
def r_resize_10_check(self, node, tensor):
    in_shape = self.shape_pick(tensor["I:out0"])
    out_shape = self.shape_pick(tensor["Resize:out0"])

    # acuity only support 3D or 4D resize
    if len(in_shape) < 2 or len(in_shape) > 4:
        return False
    # acuity only support resize width or height
    if in_shape[0] != out_shape[0] or in_shape[1] != out_shape[1]:
        return False

    return True

@rule_pyfunc_def
def r_resize_check(self, node, tensor):
    in_shape = self.shape_pick(tensor["I:out0"])
    out_shape = self.shape_pick(tensor["Resize:out0"])

    # acuity only support 3D or 4D resize
    if len(in_shape) < 2 or len(in_shape) > 4:
        return False
    # acuity only support resize width or height
    if in_shape[0] != out_shape[0] or in_shape[1] != out_shape[1]:
        return False

    unsuppored_trans_mode = [
        #'pytorch_half_pixel',
        #for pytorch_half_pixel, we assue that length_resized will always > 1,
        #in this condition, it equals to 'half_pixel',
        #but if length_resized == 1, there will be some precision issue
        'tf_half_piexl_for_nn',
        'tf_crop_and_resize'
    ]
    trans_mode = self.attr_pick(node['Resize'], 'coordinate_transformation_mode', 'half_pixel')
    if trans_mode in unsuppored_trans_mode:
        return False

    mode = self.attr_pick(node['Resize'], 'mode', 'nearest')
    nearest_mode = self.attr_pick(node['Resize'], 'nearest_mode', 'round_prefer_floor')
    if mode == 'nearest' and 'ceil' in nearest_mode:
        return False

    # pytorch coeff_a is -0.75
    # tf coeff_a is -0.5, we only support this coeff_a
    coeff_a = self.attr_pick(node['Resize'], 'cubic_coeff_a', -0.75)
    if mode == 'cubic' and coeff_a != -0.5:
        return False

    return True

@rule_pyfunc_def
def r_resize_get_new_size(self, node, tensor):
    out_shape = self.shape_pick(tensor["Resize:out0"])
    new_size = out_shape[2:] # [batch, channel, height, width] or [batch, channel, width]
    return new_size

@rule_pyfunc_def
def r_resize_get_type(self, node, tensor):
    mode = self.attr_pick(node['Resize'], 'mode', 'nearest').lower()
    _mode_map = {
        "nearest": "nearest",
        "linear": "bilinear",
        "cubic": "bicubic"
    }

    _maped_mode = "nearest"
    if mode in _mode_map.keys():
        _maped_mode = _mode_map[mode]
    return _maped_mode

@rule_pyfunc_def
def r_resize_get_align_corners(self, node, tensor):
    trans_mode = self.attr_pick(node['Resize'], 'coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        return True
    return False

@rule_pyfunc_def
def r_resize_get_half_pixel(self, node, tensor):
    trans_mode = self.attr_pick(node['Resize'], 'coordinate_transformation_mode', 'half_pixel')
    # for pytorch_half_pixel, we assue that length_resized will always > 1,
    # in this condition, it equals to 'half_pixel',
    # but if length_resized == 1, there will be some precision issue
    if trans_mode in ['half_pixel', 'pytorch_half_pixel']:
        return True
    return False

@rule_pyfunc_def
def r_mm_check_wb(self, node, tensor, in_tensor, weight, bias = None):
    input_shape = self.shape_pick(tensor[in_tensor])
    weight_shape = self.shape_pick(tensor[weight])
    if len(input_shape) != 2 or len(weight_shape) != 2:
        return False

    if bias is not None:
        bias_shape = self.shape_pick(tensor[bias])
        weights = self.shape_pick(tensor[weight])[1]
        if len(bias_shape) != 1 or weights != bias_shape[0]:
            return False
    return True

r_variable = {
"ruler_name": "r_variable",
"src_ops_alias": ["Constant"],
"src_inter_flow": [],
"src_in_anchor": [],
"src_out_tensor": ["Constant:out0"],
"acu_lys_alias": ["variable"],
"src_acu_in_tensor_map": [],
"src_acu_out_tensor_map": [["Constant:out0", "variable:out0"]],
"param_map": {"variable": {'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['Constant:out0'])"],
                           'is_scalar': ['BOOL', 'CODE',
                           "True if len(self.tensor_to_numpy_without_convert_0darry(tensor['Constant:out0']).shape) "
                           "== 0 else False "]}},
"blob_map": {"variable": {'data':
                              ['CODE',
                               "np.array([self.tensor_to_numpy(tensor['Constant:out0'])]) "\
                               " if self.tensor_to_numpy(tensor['Constant:out0']).shape == () "\
                               "else self.tensor_to_numpy(tensor['Constant:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_variable)

r_rsp_mm_add = {
"ruler_name": "r_rsp_mm_add",
"src_ops_alias": ["Reshape", "MatMul", "Add", "Constant_0", "Constant_1"],
"src_inter_flow":
    [["Reshape:out0", "MatMul:in0"], ["MatMul:out0", "Add:in0"], ["Constant_0:out0", "MatMul:in1"],
     ["Constant_1:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='Reshape:out0', weight='Constant_0:out0', bias='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
ruler_list.append(r_rsp_mm_add)

r_rsp_mm_add_v5 = {
"ruler_name": "r_rsp_mm_add_v5",
"src_ops_alias": ["Reshape", "MatMul", "Add", "Constant_0", "Constant_1", "Constant_2"],
"src_inter_flow":
    [["Reshape:out0", "MatMul:in0"], ["MatMul:out0", "Add:in0"], ["Constant_0:out0", "MatMul:in1"],
     ["Constant_1:out0", "Add:in1"], ["Constant_2:out0", "Reshape:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map":
    {"fullconnect":
         {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
          "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"],
          "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='Reshape:out0', weight='Constant_0:out0', bias='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_mm_add_v5)

r_mm_add = {
"ruler_name": "r_mm_add",
"src_ops_alias": ["MatMul", "Add", "Constant_0", "Constant_1"],
"src_inter_flow": [["MatMul:out0", "Add:in0"], ["Constant_0:out0", "MatMul:in1"], ["Constant_1:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='I:out0', weight='Constant_0:out0', bias='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_mm_add)

r_mm = {
"ruler_name": "r_mm",
"src_ops_alias": ["MatMul", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "MatMul:in1"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["MatMul:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["MatMul:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
                   "bias": ["BOOL", "VALUE", False]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='I:out0', weight='Constant_0:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_mm)

r_gemm = {
"ruler_name": "r_gemm",
"src_ops_alias": ["Gemm"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Gemm:in0"], ['I_1:out0', "Gemm:in1"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map":[["I:out0", "matmul:in0"], ['I_1:out0', "matmul:in1"]],
"src_acu_out_tensor_map": [["Gemm:out0", "matmul:out0"]],
"param_map":{
    "matmul":{
        'transpose_a': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transA', 0) == 0 else True"],
        'transpose_b': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transB', 0) == 0 else True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm)

r_gemm_2_fc = {
"ruler_name": "r_gemm_2_fc",
"src_ops_alias": ["Gemm", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Gemm:in1"]],
"src_in_anchor": [["I:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
                   "bias": ["BOOL", "VALUE", "False"]}},
"blob_map": {"fullconnect": {"weight": ["CODE", "self.fc_weight(tensor['Constant_0:out0'], node['Gemm'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 2 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc)

r_gemm_2_fc_wb = {
"ruler_name": "r_gemm_2_fc_wb",
"src_ops_alias": ["Gemm", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Gemm:in1"], ["Constant_1:out0", "Gemm:in2"]],
"src_in_anchor": [["I:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight": ["CODE",
                     "self.gemm_weight_blob(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant_1:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc_wb)

r_gemm_2_fc_wb_notranspose = {
"ruler_name": "gemm_2_fc_notranspose",
"src_ops_alias": ["Gemm", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Gemm:in1"], ["Constant_1:out0", "Gemm:in2"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {
        "fullconnect":{
            "weights": ["INT", "CODE", "self.gemm_weights_param(tensor['Constant:out0'], node['Gemm'], 'transB')"],
            "bias": ["BOOL", "VALUE", True]
        }
    },
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant_1:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and"\
    " self.attr_pick(node['Gemm'], 'transB', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc_wb_notranspose)

r_gemm_2_fc_4d_wb_notranspose = {
"ruler_name": "r_gemm_2_fc_4d_wb_notranspose",
"src_ops_alias": ["Gemm", "Reshape", "Reshape_1", "Constant", "Constant_1"],
"src_inter_flow": [["Reshape:out0", "Gemm:in0"], ["Reshape_1:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"],
                   ["Constant_1:out0", "Reshape_1:in0"]],
"src_in_anchor": [["I_0:out0", "Reshape:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Constant:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Reshape_1:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and"\
    " self.attr_pick(node['Gemm'], 'transB', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_gemm_2_fc_4d_wb_notranspose)

r_gemm_2_fc_4d_wb_notranspose_v5 = {
"ruler_name": "r_gemm_2_fc_4d_wb_notranspose_v5",
"src_ops_alias": ["Gemm", "Reshape", "Reshape_1", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Reshape:out0", "Gemm:in0"], ["Reshape_1:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"],
                   ["Constant_1:out0", "Reshape:in1"], ["Constant_2:out0", "Reshape_1:in0"],
                   ["Constant_3:out0", "Reshape_1:in1"]],
"src_in_anchor": [["I_0:out0", "Reshape:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Reshape_1:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant_2:out0'], node['Gemm'], 'transB', \
                    self.shape_pick(tensor['Reshape_1:out0']))"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and"\
    " self.attr_pick(node['Gemm'], 'transB', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_gemm_2_fc_4d_wb_notranspose_v5)

r_gemm_2_fc_wb_bc = {
"ruler_name": "r_gemm_2_fc_wb_bc",
"src_ops_alias": ["Gemm", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Gemm:in1"], ["Constant_1:out0", "Gemm:in2"]],
"src_in_anchor": [["I:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight": ["CODE", "self.gemm_weight_blob(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "np.ones(self.shape_pick(tensor['Constant_0:out0'])[0], dtype=np.float32)*\
               self.tensor_to_numpy(tensor['Constant_1:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1 and "\
    "self.shape_pick(tensor['Constant_1:out0'])[0] == 1 ",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc_wb_bc)

r_fullconnect_with_wbrsp = {
"ruler_name": "r_fullconnect_with_wbrsp",
"src_ops_alias": ["Gemm", "Reshape", "Constant", "Constant_1"],
"src_inter_flow": [["Reshape:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"], ["Constant_1:out0", "Reshape:in0"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Reshape:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant_1:out0'], node['Gemm'], 'transB', \
                    self.shape_pick(tensor['Reshape:out0']))"
               ],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
#Gemm:Gemm_141;Reshape:Reshape_140;Constant:Initializer_115;Constant_1:Initializer_114
ruler_list.append(r_fullconnect_with_wbrsp)

r_fullconnect_with_weight_at_in0_with_reshape = {
"ruler_name": "fc_w@in0_t@in1_with_reshape",
"src_ops_alias": ["Gemm", "Reshape", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Reshape:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"],
                   ["Constant_1:out0", "Reshape:in0"], ["Constant_2:out0", "Reshape:in1"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Reshape:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant_1:out0'], node['Gemm'], 'transB', \
                    self.shape_pick(tensor['Reshape:out0']))"
               ],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
#Gemm:Gemm_142;Reshape:Reshape_141;Constant:Initializer_114;Constant_1:Initializer_115;Constant_2:Initializer_117
ruler_list.append(r_fullconnect_with_weight_at_in0_with_reshape)


r_tanh = {
"ruler_name": "r_tanh",
"src_ops_alias": ["Tanh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Tanh:in0"]],
"src_out_tensor": ["Tanh:out0"],
"acu_lys_alias": ["tanh"],
"src_acu_in_tensor_map": [["I:out0", "tanh:in0"]],
"src_acu_out_tensor_map": [["Tanh:out0", "tanh:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_tanh)

r_relu = {
"ruler_name": "r_relu",
"src_ops_alias": ["Relu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Relu:in0"]],
"src_out_tensor": ["Relu:out0"],
"acu_lys_alias": ["relu"],
"src_acu_in_tensor_map": [["I:out0", "relu:in0"]],
"src_acu_out_tensor_map": [["Relu:out0", "relu:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_relu)

r_elu = {
"ruler_name": "r_elu",
"src_ops_alias": ["Elu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Elu:in0"]],
"src_out_tensor": ["Elu:out0"],
"acu_lys_alias": ["elu"],
"src_acu_in_tensor_map": [["I:out0", "elu:in0"]],
"src_acu_out_tensor_map": [["Elu:out0", "elu:out0"]],
"param_map": {"elu": {"alpha": ["FLOAT", "CODE", "self.attr_pick(node['Elu'], 'alpha', 1.0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_elu)

r_celu = {
"ruler_name": "r_celu",
"src_ops_alias": ["Celu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Celu:in0"]],
"src_out_tensor": ["Celu:out0"],
"acu_lys_alias": ["celu"],
"src_acu_in_tensor_map": [["I:out0", "celu:in0"]],
"src_acu_out_tensor_map": [["Celu:out0", "celu:out0"]],
"param_map": {"celu": {"alpha": ["FLOAT", "CODE", "self.attr_pick(node['Celu'], 'alpha', 1.0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [12, -1]}
ruler_list.append(r_celu)

r_sigmoid = {
"ruler_name": "r_sigmoid",
"src_ops_alias": ["Sigmoid"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sigmoid:in0"]],
"src_out_tensor": ["Sigmoid:out0"],
"acu_lys_alias": ["Sigmoid"],
"src_acu_in_tensor_map": [["I:out0", "Sigmoid:in0"]],
"src_acu_out_tensor_map": [["Sigmoid:out0", "Sigmoid:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_sigmoid)

r_hard_sigmoid = {
"ruler_name": "r_hard_sigmoid",
"src_ops_alias": ["HardSigmoid"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "HardSigmoid:in0"]],
"src_out_tensor": ["HardSigmoid:out0"],
"acu_lys_alias": ["hard_sigmoid"],
"src_acu_in_tensor_map": [["I:out0", "hard_sigmoid:in0"]],
"src_acu_out_tensor_map": [["HardSigmoid:out0", "hard_sigmoid:out0"]],
"param_map": {
    "hard_sigmoid":{
        "alpha": ["FLOAT", "CODE", "self.attr_pick(node['HardSigmoid'], 'alpha', 0.2)"],
        "beta": ["FLOAT", "CODE", "self.attr_pick(node['HardSigmoid'], 'beta', 0.5)"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_hard_sigmoid)

r_leakrelu = {
"ruler_name": "r_leakrelu",
"src_ops_alias": ["LeakyRelu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LeakyRelu:in0"]],
"src_out_tensor": ["LeakyRelu:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["LeakyRelu:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {"leaky_ratio": ["FLOAT", "CODE", "self.attr_pick(node['LeakyRelu'], 'alpha', 0.01)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_leakrelu)

r_prelu_with_reshape = {
"ruler_name": "r_prelu_with_reshape",
"src_ops_alias": ["PRelu", "Reshape", "Constant", "Constant_1"],
"src_inter_flow": [["Reshape:out0", "PRelu:in1"],
                   ["Constant:out0", "Reshape:in0"],
                   ["Constant_1:out0", "Reshape:in1"]],
"src_in_anchor": [["I_0:out0", "PRelu:in0"]],
"src_out_tensor": ["PRelu:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I_0:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["PRelu:out0", "prelu:out0"]],
"acu_inter_flow": [],
"param_map": {"prelu": {}},
"blob_map": {"prelu": {"a":
              ["CODE",
               "self.prelu_alpha(tensor['I_0:out0'], tensor['Reshape:out0'])"]
          }},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_prelu_with_reshape)

r_prelu = {
"ruler_name": "r_prelu",
"src_ops_alias": ["PRelu", "Constant_0"],
"src_inter_flow": [ ["Constant_0:out0", "PRelu:in1"]],
"src_in_anchor": [["I:out0", "PRelu:in0"]],
"src_out_tensor": ["PRelu:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["PRelu:out0", "prelu:out0"]],
"param_map": {},
"blob_map": {"prelu": {"a": ["CODE", "self.prelu_alpha(tensor['I:out0'], tensor['Constant_0:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_prelu)

r_prelu_unsqueeze = {
"ruler_name": "r_prelu_unsqueeze",
"src_ops_alias": ["PRelu", "Constant", "Unsqueeze"],
"src_inter_flow": [["Constant:out0","Unsqueeze:in0"], ["Unsqueeze:out0", "PRelu:in1"]],
"src_in_anchor": [["I:out0", "PRelu:in0"]],
"src_out_tensor": ["PRelu:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["PRelu:out0", "prelu:out0"]],
"param_map":  {},
"blob_map": {"prelu": {"a": ["CODE", "self.prelu_alpha(tensor['I:out0'], tensor['Unsqueeze:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_prelu_unsqueeze)

r_reciprocal = {
"ruler_name": "r_reciprocal",
"src_ops_alias": ["Reciprocal"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Reciprocal:in0"]],
"src_out_tensor": ["Reciprocal:out0"],
"acu_lys_alias": ["variable", "Divide"],
"src_acu_in_tensor_map": [["I_0:out0", "Divide:in1"]],
"src_acu_out_tensor_map": [["Reciprocal:out0", "Divide:out0"]],
"acu_inter_flow": [["variable:out0", "Divide:in0"]],
"param_map": {"variable": {}},
"blob_map": {"variable": {'data': ['CODE', "np.array([1], dtype=np.float32)"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reciprocal)

r_pow = {
"ruler_name": "r_pow",
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
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_pow)

r_equal = {
"ruler_name": "r_equal",
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
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_equal)

r_less = {
"ruler_name": "r_less",
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
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_less)

r_less_equal = {
"ruler_name": "r_less_equal",
"src_ops_alias": ["LessOrEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LessOrEqual:in0"], ["I_1:out0", "LessOrEqual:in1"]],
"src_out_tensor": ["LessOrEqual:out0"],
"acu_lys_alias": ["less_equal"],
"src_acu_in_tensor_map": [["I:out0", "less_equal:in0"], ["I_1:out0", "less_equal:in1"]],
"src_acu_out_tensor_map": [["LessOrEqual:out0", "less_equal:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [12, -1]
}
ruler_list.append(r_less_equal)

r_conv1d = {
"ruler_name": "r_conv1d",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv1d:out0"]],
"param_map":
{
"conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", True],
"pad_method": ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None)"\
               "== None else 'padding_const'"],
"ksize": ["INT", "PYFUNC", r_conv1d_get_kernel_shape(kernel_name='Constant_0:out0')],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_conv1d)

r_group_conv1d = {
"ruler_name": "r_group_conv1d",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["group_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "group_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "group_conv1d:out0"]],
"param_map":
{
"group_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", True],
"pad_method": ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None)"\
               "== None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "group_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": r_group_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_group_conv1d)

r_depthwise_conv1d = {
"ruler_name": "r_depthwise_conv1d",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["depthwise_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "depthwise_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "depthwise_conv1d:out0"]],
"param_map":
{
"depthwise_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", True],
"pad_method": ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None)"\
               "== None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"multiplier": ["INT", "CODE",
               "int(self.shape_pick(tensor['Constant_0:out0'])[0]/self.shape_pick(tensor['I:out0'])[1])"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "depthwise_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 2,
"pre_condition": r_depthwise_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_depthwise_conv1d)

r_conv1d_no_bias = {
"ruler_name": "r_conv1d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv1d:out0"]],
"param_map":
{
"conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", False],
"pad_method": ["STRING", "CODE",
"'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"ksize": ["INT", "PYFUNC", r_conv1d_get_kernel_shape(kernel_name='Constant_0:out0')],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_conv1d_no_bias)

r_group_conv1d_no_bias = {
"ruler_name": "r_group_conv1d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["group_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "group_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "group_conv1d:out0"]],
"param_map":
{
"group_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", False],
"pad_method": ["STRING", "CODE",
"'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "group_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": r_group_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_group_conv1d_no_bias)

r_depthwise_conv1d_no_bias = {
"ruler_name": "r_depthwise_conv1d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["depthwise_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "depthwise_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "depthwise_conv1d:out0"]],
"param_map":
{
"depthwise_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", False],
"pad_method": ["STRING", "CODE",
"'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"multiplier": ["INT", "CODE",
               "int(self.shape_pick(tensor['Constant_0:out0'])[0]/self.shape_pick(tensor['I:out0'])[1])"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "depthwise_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 2,
"pre_condition": r_depthwise_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_depthwise_conv1d_no_bias)

r_conv = {
"ruler_name": "r_conv",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv)

r_conv2d_op = {
"ruler_name": "r_conv2d_op",
"src_ops_alias": ["Conv"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Conv:in0"], ["I_1:out0", "Conv:in1"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv2d_op"],
"src_acu_in_tensor_map": [["I_0:out0", "conv2d_op:in0"], ["I_1:out0", "conv2d_op:in1"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv2d_op:out0"]],
"acu_inter_flow": [],
"param_map": {
    "conv2d_op": {
    "padding":
        ["STRING",
         "CODE",
         "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
    "pad":
        ["INTS",
         "CODE",
         "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
    "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
    "dilation":
        ['INT',
         'CODE',
         "self.attr_pick(node['Conv'], 'dilations')" \
         " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)" \
         " else self.attr_pick(node['Conv'], 'dilations')[0]"],
}},
"blob_map": {"conv2d_op": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_conv2d_op)

r_depthwise_conv2d_op = {
"ruler_name": "r_depthwise_conv2d_op",
"src_ops_alias": ["Conv"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Conv:in0"], ["I_1:out0", "Conv:in1"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["depthwise_conv2d_op"],
"src_acu_in_tensor_map": [["I_0:out0", "depthwise_conv2d_op:in0"], ["I_1:out0", "depthwise_conv2d_op:in1"]],
"src_acu_out_tensor_map": [["Conv:out0", "depthwise_conv2d_op:out0"]],
"acu_inter_flow": [],
"param_map": {
    "depthwise_conv2d_op": {
    "padding":
        ["STRING",
         "CODE",
         "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
    "pad":
        ["INTS",
         "CODE",
         "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
    "pad_method":
        ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
    "ksize_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[1]"],
    "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
    "ksize_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
    "dilation":
        ['INT',
         'CODE',
         "self.attr_pick(node['Conv'], 'dilations')" \
         " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)" \
         " else self.attr_pick(node['Conv'], 'dilations')[0]"],
}},
"blob_map": {"depthwise_conv2d_op": {}},
"priority_tip": 1,
"pre_condition": "self.attr_pick(node['Conv'], 'group', 1) == self.shape_pick(tensor['I_0:out0'])[1]",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_depthwise_conv2d_op)

r_conv_nchw_squeeze = {
"ruler_name": "r_conv_nchw_squeeze",
"src_ops_alias": ["Conv", "Constant", "Squeeze", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Conv:in1"], ["Squeeze:out0", "Conv:in2"], ["Constant_1:out0", "Squeeze:in0"]],
"src_in_anchor": [["I_0:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I_0:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Squeeze:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[1]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Squeeze:out0'])"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_conv_nchw_squeeze)

r_conv_no_bias = {
"ruler_name": "r_conv_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", False],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv_no_bias)

r_conv3d_no_bias = {
"ruler_name": "r_conv3d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv3d"],
"src_acu_in_tensor_map": [["I:out0", "conv3d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv3d:out0"]],
"param_map":
{"conv3d":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", False],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[4]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"ksize_d": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[2]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[1]"],
"stride_d": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0, 0, 0]), [0, 3, 1, 4, 2, 5])]"]
}
},
"blob_map": {"conv3d":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],}},
"acu_inter_flow": [["Constant_0:out0", "conv3d:in1"]],
"priority_tip": 1,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 5",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv3d_no_bias)

r_conv_add = {
"ruler_name": "r_conv_add",
"src_ops_alias": ["Conv", "Add", "Constant_0", "Constant_1"],
"src_inter_flow": [["Conv:out0", "Add:in0"], ["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "convolution:out0"]],
"param_map":
    {"convolution":
         {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
          "pad_method":
              ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
          "bias": ["BOOL", "VALUE", True],
          "ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
          "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
          "ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
          "dilation":
             ['INT',
              'CODE',
              "self.attr_pick(node['Conv'], 'dilations')"\
              " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
              " else self.attr_pick(node['Conv'], 'dilations')[0]"],
          "padding":
              ["STRING",
               "CODE",
               "'SAME'"\
               " if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
               "'VALID' "],
          "pad":
              ["INTS",
               "CODE",
               "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
          }
     },
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.conv_bias(tensor['Constant_0:out0'], tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.is_single_refs(tensor['Conv:out0']) == True",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv_add)

r_dconvolution = {
"ruler_name": "r_dconvolution",
"src_ops_alias": ["ConvTranspose", "Constant_0", "Constant_1"],
"src_inter_flow": [ ["Constant_0:out0", "ConvTranspose:in1"], ["Constant_1:out0", "ConvTranspose:in2"]],
"src_in_anchor": [["I:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution:out0"]],
"param_map":
    {"deconvolution":
         {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant_0:out0')],
          "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
          "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
          "bias": ["BOOL", "VALUE", True],
          "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(dim='width')],
          "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
          "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(dim='height')],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
          "padding":
          ["STRING",
           "CODE",
           "'SAME'"\
           " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
           "'VALID' "],
          'pad_method':
          ["STRING",
           "CODE",
           "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
           "== 'NOTSET' else 'auto' "],
          'pad': ['INTS', 'CODE',
                  "self.array_layout(self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0]), [0, 2, 1, 3])"],
          "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {"deconvolution": {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                                 "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_dconvolution)

r_dconvolution_no_bias = {
"ruler_name": "r_dconvolution_no_bias",
"src_ops_alias": ["ConvTranspose", "Constant_0"],
"src_inter_flow": [ ["Constant_0:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution:out0"]],
"param_map":
    {"deconvolution":
         {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant_0:out0')],
          "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
          "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
          "bias": ["BOOL", "VALUE", False],
          "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(dim='width')],
          "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
          "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(dim='height')],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
          "padding":
          ["STRING",
           "CODE",
           "'SAME'"\
           " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
           "'VALID' "],
          'pad_method':
          ["STRING",
           "CODE",
           "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
           "== 'NOTSET' else 'auto' "],
          'pad': ['INTS', 'CODE',
                  "self.array_layout(self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0]), [0, 2, 1, 3])"],
          "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {"deconvolution": {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_dconvolution_no_bias)

@rule_pyfunc_def
def r_bias_var_check(self, node, tensor, bias_tensor):
    data = self.tensor_to_numpy(tensor[bias_tensor])
    if data.ndim >= 2 and data.shape[1] == data.size:
        return True
    return False

r_deconvolution_with_add = {
"ruler_name": 'r_deconvolution_with_add',
"src_ops_alias": ["Add", "ConvTranspose", "Constant", "Constant_1"],
"src_inter_flow": [["ConvTranspose:out0", "Add:in0"], \
        ["Constant_1:out0", "Add:in1"], ["Constant:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I_0:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I_0:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "deconvolution:out0"]],
"acu_inter_flow": [],
"param_map": {"deconvolution":
         {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant:out0')],
          "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
          "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
          "bias": ["BOOL", "VALUE", True],
          "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(dim='width')],
          "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
          "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(dim='height')],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
          "padding":
          ["STRING",
           "CODE",
           "'SAME'"\
           " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
           "'VALID' "],
          'pad_method':
          ["STRING",
           "CODE",
           "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
           "== 'NOTSET' else 'auto' "],
          'pad': ['INTS', 'CODE',
                  "self.array_layout(self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0]), [0, 2, 1, 3])"],
          "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {"deconvolution": {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0']).flatten()"],
    }},
"priority_tip": 0,
"pre_condition": r_bias_var_check(bias_tensor='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_deconvolution_with_add)

r_deconv1d_no_bias = {
"ruler_name": "r_dconv1d_no_bias",
"src_ops_alias": ["ConvTranspose", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution1d:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution1d:out0"]],
"acu_lys_alias": ["deconvolution1d"],
"acu_inter_flow": [],
"param_map": {"deconvolution1d": {'ksize': ['INT', 'CODE', "self.shape_pick(tensor['Constant_0:out0'])[2]"],
                           'stride': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'strides')[0]"],
                           'bias': ['BOOL', 'VALUE', False],
                           "padding":
                           ["STRING",
                            "CODE",
                            "'SAME' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in "
                            "['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
                           'pad_method':
                           ["STRING",
                            "CODE",
                            "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
                            "== 'NOTSET' else 'auto' "],
                           'pad': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], 'pads')"],
                           'group_number': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
                           'weights': ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant_0:out0')],
                           'dilation': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'dilations', [1])[0]"],
                           'output_shape': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
                           'output_padding': ['INT', 'CODE',
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0) "
                                              "if isinstance("
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0), int)"
                                              "else self.attr_pick(node['ConvTranspose'], 'output_padding', 0)[0]"]
                           }},
"blob_map": {"deconvolution1d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 3"}
ruler_list.append(r_deconv1d_no_bias)

r_deconv1d = {
"ruler_name": 'r_deconv1d',
"src_ops_alias": ["ConvTranspose", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "ConvTranspose:in1"], ["Constant_1:out0", "ConvTranspose:in2"]],
"src_in_anchor": [["I_0:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution1d"],
"src_acu_in_tensor_map": [["I_0:out0", "deconvolution1d:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution1d:out0"]],
"acu_inter_flow": [],
"param_map": {"deconvolution1d": {'ksize': ['INT', 'CODE', "self.shape_pick(tensor['Constant:out0'])[2]"],
                           'stride': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'strides')[0]"],
                           'bias': ['BOOL', 'VALUE', True],
                           "padding":
                           ["STRING",
                            "CODE",
                            "'SAME' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in "
                            "['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
                           'pad_method':
                           ["STRING",
                            "CODE",
                            "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
                            "== 'NOTSET' else 'auto' "],
                           'pad': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], 'pads')"],
                           'group_number': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
                           'weights': ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant:out0')],
                           'dilation': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'dilations', [1])[0]"],
                           'output_shape': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
                           'output_padding': ['INT', 'CODE',
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0) "
                                              "if isinstance("
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0), int)"
                                              "else self.attr_pick(node['ConvTranspose'], 'output_padding', 0)[0]"]
                           }},
"blob_map": {"deconvolution1d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
                                 'bias': ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_deconv1d)

r_bn_v6 = {
"ruler_name": "r_bn_v6",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map": {"batchnormalize":
                  {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
    {"batchnormalize":
         {"gamma": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
          "beta": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
          "variance": ["CODE", "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
          "mean": ["CODE", "self.tensor_to_numpy(tensor['Constant_3:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_bn_v6)

r_bn_v5 = {
"ruler_name": "r_bn_v5",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
{"batchnormalize":
{"gamma":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[1] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_0:out0'])"],
"beta":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[2] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_1:out0'])"],
"variance":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[3] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_2:out0'])"],
"mean":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[4] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_3:out0'])"],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_bn_v5)

r_bn_mul_add_v5 = {
"ruler_name": "r_bn_mul_add_v5",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3",
                  "Mul", "Constant_4", "Add", "Constant_5"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"],
                   ["BatchNormalization:out0", "Mul:in0"], ["Constant_4:out0", "Mul:in1"],
                   ["Mul:out0", "Add:in0"], ["Constant_5:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
{"batchnormalize":
{"gamma":
  ["CODE",
   "self.tensor_to_numpy(tensor['Constant_4:out0']) "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[1] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_0:out0']) * self.tensor_to_numpy(tensor['Constant_4:out0'])"],
"beta":
  ["CODE",
   "self.tensor_to_numpy(tensor['Constant_5:out0']) "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[1] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_1:out0']) * self.tensor_to_numpy(tensor['Constant_4:out0']) + "\
   "self.tensor_to_numpy(tensor['Constant_5:out0'])"],
"variance":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[3] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_2:out0'])"],
"mean":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[4] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_3:out0'])"],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_bn_mul_add_v5)

r_bn_mul_add_v6 = {
"ruler_name": "r_bn_mul_add_v6",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3",
                  "Mul", "Constant_4", "Add", "Constant_5"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"],
                   ["BatchNormalization:out0", "Mul:in0"], ["Constant_4:out0", "Mul:in1"],
                   ["Mul:out0", "Add:in0"], ["Constant_5:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
{"batchnormalize":
{"gamma":
  ["CODE",
   " self.tensor_to_numpy(tensor['Constant_0:out0']) * self.tensor_to_numpy(tensor['Constant_4:out0'])"],
"beta":
  ["CODE",
   "self.tensor_to_numpy(tensor['Constant_1:out0']) * self.tensor_to_numpy(tensor['Constant_4:out0']) + "\
   "self.tensor_to_numpy(tensor['Constant_5:out0'])"],
"variance":
  ["CODE",
   " self.tensor_to_numpy(tensor['Constant_2:out0'])"],
"mean":
  ["CODE",
   " self.tensor_to_numpy(tensor['Constant_3:out0'])"],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_bn_mul_add_v6)

r_maxpool = {
"ruler_name": "r_maxpool",
"src_ops_alias": ["MaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MaxPool:in0"]],
"src_out_tensor": ["MaxPool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["MaxPool:out0", "pooling:out0"]],
"param_map": {"pooling":
  {"type": ["STRING", "VALUE", "MAX"],
   "pad_method": ["STRING", "CODE",
   "'auto' if self.attr_pick(node['MaxPool'], 'pads', None) == None else 'padding_const'"],
   "pad_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0])[0]"],
   "ksize_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[1]"],
   "stride_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1])[1]"],
   "ksize_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[0]"],
   "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['MaxPool'], 'ceil_mode') == 1 else 'floor'"],
   "stride_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1])[0]"],
   "padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['MaxPool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID'"
    ],
   "pad":
       ["INTS",
        "CODE",
        "[p for p in self.array_layout(self.attr_pick(node['MaxPool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
   "pad_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['MaxPool'], 'kernel_shape')) == 2",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_maxpool)

r_maxpool_3d = {
"ruler_name": "r_maxpool3d",
"src_ops_alias": ["MaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "MaxPool:in0"]],
"src_out_tensor": ["MaxPool:out0"],
"acu_lys_alias": ["pool3d"],
"src_acu_in_tensor_map": [["I_0:out0", "pool3d:in0"]],
"src_acu_out_tensor_map": [["MaxPool:out0", "pool3d:out0"]],
"acu_inter_flow": [],
"param_map": {"pool3d": {
   "type": ["STRING", "VALUE", "MAX"],
   "pad_method": ["STRING", "CODE",
   "'auto' if self.attr_pick(node['MaxPool'], 'pads', None) == None else 'padding_const'"],
   "ksize_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[0]"],
   "ksize_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[1]"],
   "ksize_d": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[2]"],
   "stride_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1, 1])[0]"],
   "stride_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1, 1])[1]"],
   "stride_d": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1, 1])[2]"],
   "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['MaxPool'], 'ceil_mode') == 1 else 'floor'"],
   "padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['MaxPool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID'"
    ],
   "pad":
       ["INTS",
        "CODE",
        "[p for p in self.array_layout(self.attr_pick(node['MaxPool'], 'pads',"
        " [ 0, 0, 0, 0, 0, 0]), [0, 2, 4, 1, 3, 5])]"],
   }},
"blob_map": {"pool3d": {}},
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['MaxPool'], 'kernel_shape')) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_maxpool_3d)

r_avgpool = {
"ruler_name": "r_avgpool",
"src_ops_alias": ["AveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AveragePool:in0"]],
"src_out_tensor": ["AveragePool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["AveragePool:out0", "pooling:out0"]],
"param_map": {"pooling":
{"type": ["STRING", "VALUE", "AVG"],
"pad_method": ["STRING", "CODE",
        "'auto' if self.attr_pick(node['AveragePool'], 'pads', None) == None else 'padding_const'"],
"pad_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0])[0]"],
"ksize_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[1]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1])[1]"],
"ksize_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[0]"],
"round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['AveragePool'], 'ceil_mode') == 1 else 'floor'"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1])[0]"],
"padding":
["STRING",
"CODE",
"'SAME' if self.attr_pick(node['AveragePool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID'"
],
"pad":
   ["INTS",
    "CODE",
    "[str(p) for p in self.array_layout(self.attr_pick(node['AveragePool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"
    ],
"pad_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['AveragePool'], 'kernel_shape')) == 2",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_avgpool)

r_avgpool1d = {
"ruler_name": "r_avgpool1d",
"src_ops_alias": ["AveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AveragePool:in0"]],
"src_out_tensor": ["AveragePool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["AveragePool:out0", "pool1d:out0"]],
"param_map": {"pool1d":
{"type": ["STRING", "VALUE", "AVG"],
"pad_method": ["STRING", "CODE",
        "'padding_const' if self.attr_pick(node['AveragePool'], 'auto_pad', 'NOTSET') == 'NOTSET' else 'auto'"],
"stride": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1])[0]"],
"ksize": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[0]"],
"round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['AveragePool'], 'ceil_mode') == 1 else 'floor'"],
"padding":
["STRING",
"CODE",
"'SAME' if self.attr_pick(node['AveragePool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID'"
],
"pad":
   ["INTS",
    "CODE",
    "[str(p) for p in self.array_layout(self.attr_pick(node['AveragePool'], 'pads', [ 0, 0]), [0, 1])]"
    ],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['AveragePool'], 'kernel_shape')) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_avgpool1d)

r_avgpool3d = {
"ruler_name": "r_avgpool3d",
"src_ops_alias": ["AveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "AveragePool:in0"]],
"src_out_tensor": ["AveragePool:out0"],
"acu_lys_alias": ["pool3d"],
"src_acu_in_tensor_map": [["I_0:out0", "pool3d:in0"]],
"src_acu_out_tensor_map": [["AveragePool:out0", "pool3d:out0"]],
"acu_inter_flow": [],
"param_map": {"pool3d": {
    "type": ["STRING", "VALUE", "AVG"],
    "pad_method": ["STRING", "CODE",
        "'auto' if self.attr_pick(node['AveragePool'], 'pads', None) == None else 'padding_const'"],
    "pad_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0, 0, 0])[0]"],
    "pad_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0, 0, 0])[1]"],
    "pad_d": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0, 0, 0])[2]"],
    "ksize_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[0]"],
    "ksize_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[1]"],
    "ksize_d": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[2]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1, 1])[0]"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1, 1])[1]"],
    "stride_d": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1, 1])[2]"],
    "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['AveragePool'], 'ceil_mode') == 1 else 'floor'"],
    "padding":
    ["STRING", "CODE",
"'SAME' if self.attr_pick(node['AveragePool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID'"],
    "pad": ["INTS", "CODE", "[str(p) for p in self.array_layout(self.attr_pick(node['AveragePool'], "
                            "'pads', [ 0, 0, 0, 0, 0, 0]), [0, 2, 4, 1, 3, 5])]"],
    }},
"blob_map": {"pool3d": {}},
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['AveragePool'], 'kernel_shape')) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_avgpool3d)

r_global_avgpool = {
"ruler_name": "r_global_avgpool",
"src_ops_alias": ["GlobalAveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalAveragePool:in0"]],
"src_out_tensor": ["GlobalAveragePool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["GlobalAveragePool:out0", "pooling:out0"]],
"param_map": {"pooling":
{"type": ["STRING", "VALUE", "AVG"],
"pad_method": ["STRING", "VALUE", "padding_const"],
"pad_h": ["INT", "CODE", "self.attr_pick(node['GlobalAveragePool'], 'pads', [0, 0, 0, 0])[0]"],
"round_type": ["STRING", "VALUE", "floor"],
"pad":
   ["INTS",
    "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalAveragePool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"
    ],
"padding": ["STRING", "VALUE", "VALID"],
"global_pooling": ["BOOL", "VALUE", True],
"pad_w": ["INT", "CODE", "self.attr_pick(node['GlobalAveragePool'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_avgpool)

r_global_avgpool1d = {
"ruler_name": "r_global_avgpool1d",
"src_ops_alias": ["GlobalAveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalAveragePool:in0"]],
"src_out_tensor": ["GlobalAveragePool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["GlobalAveragePool:out0", "pool1d:out0"]],
"param_map": {"pool1d":
{"type": ["STRING", "VALUE", "AVG"],
"pad_method": ["STRING", "VALUE", "padding_const"],
"round_type": ["STRING", "VALUE", "floor"],
"pad":
   ["INTS",
    "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalAveragePool'], 'pads', [ 0, 0]), [0, 1])]"
    ],
"padding": ["STRING", "VALUE", "VALID"],
"global_pooling": ["BOOL", "VALUE", True]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_avgpool1d)

r_global_maxpool = {
"ruler_name": "r_global_maxpool",
"src_ops_alias": ["GlobalMaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalMaxPool:in0"]],
"src_out_tensor": ["GlobalMaxPool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["GlobalMaxPool:out0", "pooling:out0"]],
"param_map": {
    "pooling":{
        "type": ["STRING", "VALUE", "MAX"],
        "pad_method": ["STRING", "VALUE", "padding_const"],
        "pad_h": ["INT", "CODE", "self.attr_pick(node['GlobalMaxPool'], 'pads', [0, 0, 0, 0])[0]"],
        "round_type": ["STRING", "VALUE", "floor"],
        "pad": ["INTS", "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalMaxPool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
        "padding": ["STRING", "VALUE", "VALID"],
        "global_pooling": ["BOOL", "VALUE", True],
        "pad_w": ["INT", "CODE", "self.attr_pick(node['GlobalMaxPool'], 'pads', [0, 0, 0, 0])[1]"]
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_maxpool)

r_global_maxpool1d = {
"ruler_name": "r_global_maxpool1d",
"src_ops_alias": ["GlobalMaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalMaxPool:in0"]],
"src_out_tensor": ["GlobalMaxPool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["GlobalMaxPool:out0", "pool1d:out0"]],
"param_map": {
    "pool1d":{
        "type": ["STRING", "VALUE", "MAX"],
        "pad_method": ["STRING", "VALUE", "padding_const"],
        "round_type": ["STRING", "VALUE", "floor"],
        "pad": ["INTS", "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalMaxPool'], 'pads', [ 0, 0]), [0, 1])]"],
        "padding": ["STRING", "VALUE", "VALID"],
        "global_pooling": ["BOOL", "VALUE", True],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_maxpool1d)

r_rsp_v1 = {
"ruler_name": "r_rsp_v1",
"src_ops_alias": ["Reshape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape":
                  {"shape":
                       ["STRING", "CODE", "self.shape_pick(tensor['Reshape:out0'])"]
                   }
              },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
ruler_list.append(r_rsp_v1)

r_rsp_v5 = {
"ruler_name": "r_rsp_v5",
"src_ops_alias": ["Reshape", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Reshape:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape": {"shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_v5)

r_rsp_v5x = {
"ruler_name": "r_rsp_v5x",
"src_ops_alias": ["Reshape", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Reshape:in0"], ["Constant_1:out0", "Reshape:in1"]],
"src_in_anchor": [],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["variable"],
"src_acu_in_tensor_map": [],
"src_acu_out_tensor_map": [["Reshape:out0", "variable:out0"]],
"param_map": {"variable": {"shape": ['ORIGIN', 'CODE', "self.shape_pick(tensor['Reshape:out0'])"]}},
"blob_map": {"variable": {'data':
                              ['CODE',
                               "np.reshape("\
                               "self.tensor_to_numpy(tensor['Constant_0:out0']), "\
                               "self.tensor_to_numpy(tensor['Constant_1:out0']).astype(np.int32).tolist())"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_v5x)

r_dynamic_rsp_5x = {
"ruler_name": "r_dynamic_rsp_5x",
"src_ops_alias": ["Reshape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Reshape:in0"], ["I_1:out0", "Reshape:in1"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape": {"shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['I_1:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_dynamic_rsp_5x)

r_squeeze_with_constant = {
"ruler_name": "r_squeeze_with_constant",
"src_ops_alias": ["Squeeze", "Constant_0"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "reshape:out0"]],
"param_map":
{"reshape":
 {"shape":
  ["INTS",
   "CODE",
   "self.squeeze_shapes(self.attr_pick(node['Squeeze'], 'axes', None), self.shape_pick(tensor['Constant_0:out0']))"
   ]
  }
 },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_squeeze_with_constant)

r_squeeze = {
"ruler_name": "r_squeeze",
"src_ops_alias": ["Squeeze"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["squeeze"],
"src_acu_in_tensor_map": [["I_0:out0", "squeeze:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "squeeze:out0"]],
"acu_inter_flow": [],
"param_map":
{"squeeze":
 {"axis_list": ["ORIGIN", "CODE", "self.attr_pick(node['Squeeze'], 'axes', None)"],
  }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_squeeze)

r_squeeze_with_constant_axes = {
"ruler_name": "r_squeeze_with_constant_axes",
"src_ops_alias": ["Squeeze", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Squeeze:in1"]],
"src_in_anchor": [["I_0:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["squeeze"],
"src_acu_in_tensor_map": [["I_0:out0", "squeeze:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "squeeze:out0"]],
"param_map":
{"squeeze":
 {"axis_list": ["ORIGIN", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0']).tolist()"],
  }
 },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_squeeze_with_constant_axes)

r_unsqueeze = {
"ruler_name": "r_unsqueeze",
"src_ops_alias": ["Unsqueeze"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Unsqueeze:in0"]],
"src_out_tensor": ["Unsqueeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I_0:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Unsqueeze:out0", "reshape:out0"]],
"acu_inter_flow": [],
"param_map":
{"reshape":
 {"shape":
  ["INTS",
   "CODE",
   "self.unsqueeze_shape(self.attr_pick(node['Unsqueeze'], 'axes'), self.shape_pick(tensor['I_0:out0']))"
   ]
  }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_unsqueeze)

r_unsqueeze_with_constant_axes = {
"ruler_name": "r_unsqueeze_with_constant_axes",
"src_ops_alias": ["Unsqueeze", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Unsqueeze:in1"]],
"src_in_anchor": [["I_0:out0", "Unsqueeze:in0"]],
"src_out_tensor": ["Unsqueeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I_0:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Unsqueeze:out0", "reshape:out0"]],
"acu_inter_flow": [],
"param_map":
{"reshape":
 {"shape":
  ["INTS",
   "CODE",
   "self.unsqueeze_shape(self.tensor_to_numpy(tensor['Constant_0:out0']).tolist(), "
   "self.shape_pick(tensor['I_0:out0']))"
   ]
  }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_unsqueeze_with_constant_axes)

{
"ruler_name": "unsqueezex",
"src_ops_alias": ["Unsqueeze"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Unsqueeze:in0"]],
"src_out_tensor": ["Unsqueeze:out0"],
"acu_lys_alias": ["unsqueeze"],
"src_acu_in_tensor_map": [["I_0:out0", "unsqueeze:in0"]],
"src_acu_out_tensor_map": [["Unsqueeze:out0", "unsqueeze:out0"]],
"acu_inter_flow": [],
"param_map": {"unsqueeze": {}},
"blob_map": {"unsqueeze": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}

r_flatten = {
"ruler_name": "r_flatten",
"src_ops_alias": ["Flatten"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Flatten:in0"]],
"src_out_tensor": ["Flatten:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Flatten:out0", "reshape:out0"]],
"param_map": {"reshape": {"shape": ["INTS", "VALUE", [0, -1]]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_flatten)

r_transpose = {
"ruler_name": "r_transpose",
"src_ops_alias": ["Transpose"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Transpose:in0"]],
"src_out_tensor": ["Transpose:out0"],
"acu_lys_alias": ["permute"],
"src_acu_in_tensor_map": [["I:out0", "permute:in0"]],
"src_acu_out_tensor_map": [["Transpose:out0", "permute:out0"]],
"param_map":
    {"permute":
         {"perm": ["STRING", "PYFUNC", r_permute_value()]}
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_transpose)

r_softmax = {
"ruler_name": "r_softmax",
"src_ops_alias": ["Softmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Softmax:in0"]],
"src_out_tensor": ["Softmax:out0"],
"acu_lys_alias": ["softmax"],
"src_acu_in_tensor_map": [["I:out0", "softmax:in0"]],
"src_acu_out_tensor_map": [["Softmax:out0", "softmax:out0"]],
"param_map": {
"softmax": {"sf_axis": ['INT', 'PYFUNC', r_softmax_get_sf_axis()]}
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_softmax_pre_cond(),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_softmax)

r_log_softmax = {
"ruler_name": "r_log_softmax",
"src_ops_alias": ["LogSoftmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LogSoftmax:in0"]],
"src_out_tensor": ["LogSoftmax:out0"],
"acu_lys_alias": ["log_softmax"],
"src_acu_in_tensor_map": [["I:out0", "log_softmax:in0"]],
"src_acu_out_tensor_map": [["LogSoftmax:out0", "log_softmax:out0"]],
"acu_inter_flow": [],
"param_map": {
"log_softmax": {"sf_axis": ['INT', 'PYFUNC', r_softmax_get_log_sf_axis()]}
},
"blob_map": {"log_softmax": {}},
"priority_tip": 0,
"pre_condition": r_logsoftmax_pre_cond(),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_log_softmax)

r_softsign = {
"ruler_name": "r_softsign",
"src_ops_alias": ["Softsign"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Softsign:in0"]],
"src_out_tensor": ["Softsign:out0"],
"acu_lys_alias": ["abs", "add", "divide", "variable"],
"src_acu_in_tensor_map": [["I_0:out0", "abs:in0"], ["I_0:out0", "divide:in0"]],
"src_acu_out_tensor_map": [["Softsign:out0", "divide:out0"]],
"acu_inter_flow": [["variable:out0", "add:in0"], ["abs:out0", "add:in1"], ["add:out0", "divide:in1"]],
"param_map": {},
"blob_map": {"variable": {'data': ['CODE', "np.ones([1], dtype=np.float32)"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_softsign)

r_dropout_out2_as_dropout_1_6 = {
"ruler_name": 'dropout_out2_as_dropout_1_6',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I_0:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"acu_inter_flow": [],
"param_map": {'dropout':{'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"]}},
"blob_map": {"dropout": {}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) == 0 and "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out2_as_dropout_1_6)

r_dropout_out2_as_noop_1_6 = {
"ruler_name": 'dropout_out2_as_noop_1_6',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) != 0 or "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out2_as_noop_1_6)

r_dropout_out1_as_dropout_1_6 = {
"ruler_name": "r_dropout_out1_as_dropout_1_6",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"param_map": {'dropout':{'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) == 0 and "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out1_as_dropout_1_6)

r_dropout_out1_as_noop_1_6 = {
"ruler_name": "r_dropout_out1_as_noop_1_6",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) != 0 or "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out1_as_noop_1_6)

r_dropout_out2_as_dropout_7 = {
"ruler_name": 'dropout_out2_as_dropout_7',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I_0:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"acu_inter_flow": [],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {"dropout": {}},
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [7, -1]}
ruler_list.append(r_dropout_out2_as_dropout_7)

r_dropout_out2_as_noop_7 = {
"ruler_name": 'dropout_out2_as_noop_7',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {"dropout": {}},
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [7, -1]}
ruler_list.append(r_dropout_out2_as_noop_7)

r_dropout_out1_as_dropout_7 = {
"ruler_name": "r_dropout_out1_as_dropout_7",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [7, -1]}
ruler_list.append(r_dropout_out1_as_dropout_7)

r_dropout_out1_as_noop_7 = {
"ruler_name": "r_dropout_out1_as_noop_7",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [7, -1]}
ruler_list.append(r_dropout_out1_as_noop_7)

r_lrn = {
"ruler_name": "r_lrn",
"src_ops_alias": ["LRN"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LRN:in0"]],
"src_out_tensor": ["LRN:out0"],
"acu_lys_alias": ["localresponsenormalization"],
"src_acu_in_tensor_map": [["I:out0", "localresponsenormalization:in0"]],
"src_acu_out_tensor_map": [["LRN:out0", "localresponsenormalization:out0"]],
"param_map":
    {"localresponsenormalization":
         {"beta": ["FLOAT", "CODE", "self.attr_pick(node['LRN'], 'beta', 0.75)"],
          "type": ["STRING", "VALUE", "NORM_ACROSS_CHANNELS"],
          "local_size": ["INT", "CODE", "self.attr_pick(node['LRN'], 'size')"],
          "alpha": ["FLOAT", "CODE", "self.attr_pick(node['LRN'], 'alpha', 1e-4)"],
          "bias": ["FLOAT", "CODE", "self.attr_pick(node['LRN'], 'bias', 1.0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_lrn)

r_reducel2_div_2_l2normalize = {
"ruler_name": "r_reducel2_div_2_l2normalize",
"src_ops_alias": ["Div", "ReduceL2"],
"src_inter_flow": [["ReduceL2:out0", "Div:in1"]],
"src_in_anchor": [["I_0:out0", "Div:in0"], ["I_0:out0", "ReduceL2:in0"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["l2normalize"],
"src_acu_in_tensor_map": [["I_0:out0", "l2normalize:in0"]],
"src_acu_out_tensor_map": [["Div:out0", "l2normalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "l2normalize":
        {'l2n_dim': ['INTS','CODE', "self.attr_pick(node['ReduceL2'], 'axes')"]}},
"blob_map": {"l2normalize": {}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['ReduceL2'], 'keepdims') == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducel2_div_2_l2normalize)

r_l2normalaize_scale = {
"ruler_name": "l2normalaize_scale",
"src_ops_alias": ["Mul", "Div", "Reshape", "Add", "Constant", "Constant_1",
                  "Sqrt", "Constant_2", "ReduceSum", "Pow", "Constant_3"],
"src_inter_flow": [["Div:out0", "Mul:in0"], ["Reshape:out0", "Mul:in1"],
                   ["Add:out0", "Div:in1"], ["Constant:out0", "Reshape:in0"], ["Constant_1:out0", "Reshape:in1"],
                   ["Sqrt:out0", "Add:in0"], ["Constant_2:out0", "Add:in1"], ["ReduceSum:out0", "Sqrt:in0"],
                   ["Pow:out0", "ReduceSum:in0"], ["Constant_3:out0", "Pow:in1"]],
"src_in_anchor": [["I_0:out0", "Div:in0"], ["I_0:out0", "Pow:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["l2normalizescale"],
"src_acu_in_tensor_map": [["I_0:out0", "l2normalizescale:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "l2normalizescale:out0"]],
"acu_inter_flow": [],
"param_map": {"l2normalizescale": {'l2n_dim': ['ORIGIN','VALUE', -1]}},
"blob_map":
    {"l2normalizescale":
         {'scale':
              ['CODE',
               "np.reshape("\
               "self.tensor_to_numpy(tensor['Constant:out0']), "\
               "self.array_layout("\
               "self.tensor_to_numpy(tensor['Constant_1:out0']).astype(np.int32).tolist(),"\
               " [0, 2, 3, 1]))"],}},
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['Constant_3:out0']) == 2.0).all() and "\
                 " self.attr_pick(node['ReduceSum'], 'axes') == [1]",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_l2normalaize_scale)

r_instancenorm = {
"ruler_name": "r_instancenorm",
"src_ops_alias": ["InstanceNormalization", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "InstanceNormalization:in1"],
                   ["Constant_1:out0", "InstanceNormalization:in2"]],
"src_in_anchor": [["I:out0", "InstanceNormalization:in0"]],
"src_out_tensor": ["InstanceNormalization:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["InstanceNormalization:out0", "instancenormalize:out0"]],
"param_map":{
    "instancenormalize":{
        'eps': ['FLOAT', 'CODE', "self.attr_pick(node['InstanceNormalization'], 'epsilon', 1e-5)"],
        'axis':['INTS', 'CODE', "list(range(2, len(self.attr_pick(node['InstanceNormalization'], '_out_shape')[0])))"],
    }
},
"blob_map": {"instancenormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['Constant_0:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_instancenorm)

r_add = {
"ruler_name": "r_add",
"src_ops_alias": ["Add"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Add:in0"], ['I_1:out0', "Add:in1"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map":[["I:out0", "add:in0"], ['I_1:out0', "add:in1"]],
"src_acu_out_tensor_map": [["Add:out0", "add:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_add)

r_sub = {
"ruler_name": "r_sub",
"src_ops_alias": ["Sub"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sub:in0"], ['I_1:out0', "Sub:in1"]],
"src_out_tensor": ["Sub:out0"],
"acu_lys_alias": ["subtract"],
"src_acu_in_tensor_map":[["I:out0", "subtract:in0"], ['I_1:out0', "subtract:in1"]],
"src_acu_out_tensor_map": [["Sub:out0", "subtract:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_sub)

r_mul = {
"ruler_name": "r_mul",
"src_ops_alias": ["Mul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mul:in0"], ['I_1:out0', "Mul:in1"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["multiply"],
"src_acu_in_tensor_map":[["I:out0", "multiply:in0"], ['I_1:out0', "multiply:in1"]],
"src_acu_out_tensor_map": [["Mul:out0", "multiply:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mul)

r_mul_with_same_input = {
"ruler_name": "r_mul_with_same_input",
"src_ops_alias": ["Mul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mul:in0"], ['I:out0', "Mul:in1"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["multiply"],
"src_acu_in_tensor_map":[["I:out0", "multiply:in0"], ['I:out0', "multiply:in1"]],
"src_acu_out_tensor_map": [["Mul:out0", "multiply:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mul_with_same_input)

r_div = {
"ruler_name": "r_div",
"src_ops_alias": ["Div"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Div:in0"], ['I_1:out0', "Div:in1"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["real_div"],
"src_acu_in_tensor_map":[["I:out0", "real_div:in0"], ['I_1:out0', "real_div:in1"]],
"src_acu_out_tensor_map": [["Div:out0", "real_div:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_div)

r_mod = {
"ruler_name": "r_mod",
"src_ops_alias": ["Mod"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mod:in0"], ['I_1:out0', "Mod:in1"]],
"src_out_tensor": ["Mod:out0"],
"acu_lys_alias": ["mod"],
"src_acu_in_tensor_map":[["I:out0", "mod:in0"], ['I_1:out0', "mod:in1"]],
"src_acu_out_tensor_map": [["Mod:out0", "mod:out0"]],
"param_map": {"mod":{"fmod":['INT', 'CODE', "self.attr_pick(node['Mod'], 'fmod', 0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mod)

r_logical_or = {
"ruler_name": "r_logical_or",
"src_ops_alias": ["Or"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Or:in0"], ['I_1:out0', "Or:in1"]],
"src_out_tensor": ["Or:out0"],
"acu_lys_alias": ["logical_or"],
"src_acu_in_tensor_map":[["I:out0", "logical_or:in0"], ['I_1:out0', "logical_or:in1"]],
"src_acu_out_tensor_map": [["Or:out0", "logical_or:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_logical_or)

r_logical_and = {
"ruler_name": "r_logical_and",
"src_ops_alias": ["And"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "And:in0"], ['I_1:out0', "And:in1"]],
"src_out_tensor": ["And:out0"],
"acu_lys_alias": ["logical_and"],
"src_acu_in_tensor_map":[["I:out0", "logical_and:in0"], ['I_1:out0', "logical_and:in1"]],
"src_acu_out_tensor_map": [["And:out0", "logical_and:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_logical_and)

r_greater = {
"ruler_name": "r_greater",
"src_ops_alias": ["Greater"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Greater:in0"], ['I_1:out0', "Greater:in1"]],
"src_out_tensor": ["Greater:out0"],
"acu_lys_alias": ["greater"],
"src_acu_in_tensor_map":[["I:out0", "greater:in0"], ['I_1:out0', "greater:in1"]],
"src_acu_out_tensor_map": [["Greater:out0", "greater:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_greater)

r_greater_equal = {
"ruler_name": "r_greater_equal",
"src_ops_alias": ["GreaterOrEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GreaterOrEqual:in0"], ['I_1:out0', "GreaterOrEqual:in1"]],
"src_out_tensor": ["GreaterOrEqual:out0"],
"acu_lys_alias": ["greater_equal"],
"src_acu_in_tensor_map":[["I:out0", "greater_equal:in0"], ['I_1:out0', "greater_equal:in1"]],
"src_acu_out_tensor_map": [["GreaterOrEqual:out0", "greater_equal:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [12, -1]}
ruler_list.append(r_greater_equal)

r_abs = {
"ruler_name": "r_abs",
"src_ops_alias": ["Abs"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Abs:in0"]],
"src_out_tensor": ["Abs:out0"],
"acu_lys_alias": ["abs"],
"src_acu_in_tensor_map": [["I:out0", "abs:in0"]],
"src_acu_out_tensor_map": [["Abs:out0", "abs:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_abs)

r_ceil = {
"ruler_name": "r_ceil",
"src_ops_alias": ["Ceil"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Ceil:in0"]],
"src_out_tensor": ["Ceil:out0"],
"acu_lys_alias": ["ceil"],
"src_acu_in_tensor_map": [["I:out0", "ceil:in0"]],
"src_acu_out_tensor_map": [["Ceil:out0", "ceil:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_ceil)

r_erf = {
"ruler_name": "r_erf",
"src_ops_alias": ["Erf"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Erf:in0"]],
"src_out_tensor": ["Erf:out0"],
"acu_lys_alias": ["erf"],
"src_acu_in_tensor_map": [["I:out0", "erf:in0"]],
"src_acu_out_tensor_map": [["Erf:out0", "erf:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_erf)

r_floor = {
"ruler_name": "r_floor",
"src_ops_alias": ["Floor"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Floor:in0"]],
"src_out_tensor": ["Floor:out0"],
"acu_lys_alias": ["floor"],
"src_acu_in_tensor_map": [["I:out0", "floor:in0"]],
"src_acu_out_tensor_map": [["Floor:out0", "floor:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_floor)

r_sqrt = {
"ruler_name": "r_sqrt",
"src_ops_alias": ["Sqrt"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sqrt:in0"]],
"src_out_tensor": ["Sqrt:out0"],
"acu_lys_alias": ["sqrt"],
"src_acu_in_tensor_map": [["I:out0", "sqrt:in0"]],
"src_acu_out_tensor_map": [["Sqrt:out0", "sqrt:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_sqrt)

r_log = {
"ruler_name": "r_log",
"src_ops_alias": ["Log"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Log:in0"]],
"src_out_tensor": ["Log:out0"],
"acu_lys_alias": ["log"],
"src_acu_in_tensor_map": [["I:out0", "log:in0"]],
"src_acu_out_tensor_map": [["Log:out0", "log:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_log)

r_cast = {
"ruler_name": "r_cast",
"src_ops_alias": ["Cast"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Cast:in0"]],
"src_out_tensor": ["Cast:out0"],
"acu_lys_alias": ["cast"],
"src_acu_in_tensor_map": [["I:out0", "cast:in0"]],
"src_acu_out_tensor_map": [["Cast:out0", "cast:out0"]],
"param_map": {
    "cast": {
        'in_data_type':["STRING", "CODE", "self.cast_map_type(node['Cast'])"],
        'out_data_type':["STRING", "CODE", "self.cast_map_type(node['Cast'])"],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_cast)

r_exp = {
"ruler_name": "r_exp",
"src_ops_alias": ["Exp"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Exp:in0"]],
"src_out_tensor": ["Exp:out0"],
"acu_lys_alias": ["exp"],
"src_acu_in_tensor_map": [["I:out0", "exp:in0"]],
"src_acu_out_tensor_map": [["Exp:out0", "exp:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_exp)

r_rsp_tsp_rsp_2_shuffle_v1 = {
"ruler_name": "r_rsp_tsp_rsp_2_shuffle_v1",
"src_ops_alias": ["Reshape", "Transpose", "Reshape_1"],
"src_inter_flow": [["Reshape:out0", "Transpose:in0"], ["Transpose:out0", "Reshape_1:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape_1:out0"],
"acu_lys_alias": ["shuffle"],
"src_acu_in_tensor_map": [["I:out0", "shuffle:in0"]],
"src_acu_out_tensor_map": [["Reshape_1:out0", "shuffle:out0"]],
"param_map": {"shuffle": {"group_number": ["INT", "CODE", "self.attr_pick(node['Reshape'], 'shape')[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['Reshape'], 'shape')) == 5 and "\
    "len(self.attr_pick(node['Transpose'], 'perm')) == 5 and "\
    "len(self.attr_pick(node['Reshape_1'], 'shape')) == 4 and "\
    "self.attr_pick(node['Transpose'], 'perm') == [0, 2, 1, 3, 4]",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
ruler_list.append(r_rsp_tsp_rsp_2_shuffle_v1)

r_rsp_tsp_rsp_2_shuffle_v5 = {
"ruler_name": "r_rsp_tsp_rsp_2_shuffle_v5",
"src_ops_alias": ["Reshape", "Transpose", "Reshape_1", "Constant_0"],
"src_inter_flow": [["Reshape:out0", "Transpose:in0"], ["Transpose:out0", "Reshape_1:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape_1:out0"],
"acu_lys_alias": ["shuffle"],
"src_acu_in_tensor_map": [["I:out0", "shuffle:in0"]],
"src_acu_out_tensor_map": [["Reshape_1:out0", "shuffle:out0"]],
"param_map": {"shuffle": {"group_number": ["INT", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Reshape'].inputs[1])) == 5 and "\
    "len(self.attr_pick(node['Transpose'], 'perm')) == 5 and "\
    "len(self.tensor_to_numpy(tensor['Reshape_1'].inputs[1])) == 4 and "\
    "self.attr_pick(node['Transpose'], 'perm') == [0, 2, 1, 3, 4]",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_tsp_rsp_2_shuffle_v5)

r_identity = {
"ruler_name": "r_identity",
"src_ops_alias": ["Identity"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Identity:in0"]],
"src_out_tensor": ["Identity:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Identity:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_identity)

r_constantfill = {
"ruler_name": "r_constantfill",
"src_ops_alias": ["ConstantFill"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ConstantFill:in0"]],
"src_out_tensor": ["ConstantFill:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ConstantFill:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_constantfill)

r_slice = {
"ruler_name": "r_slice",
"src_ops_alias": ["Slice"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"param_map":
    {"slice":
         {"begin": ["INTS", "PYFUNC", r_slice_get_begin()],
          "size": ["INTS", "PYFUNC", r_slice_get_size()],
          }
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Slice'], 'axes', None) == None",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 9]}
ruler_list.append(r_slice)

r_slice_axes = {
"ruler_name": "r_slice_axes",
"src_ops_alias": ["Slice"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"param_map":
    {"slice":
         {"begin": ["INTS", "PYFUNC", r_slice_get_begin()],
          "size": ["INTS", "PYFUNC", r_slice_get_size()],
          }
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Slice'], 'axes', None) != None",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 9]}
ruler_list.append(r_slice_axes)

r_slice_ex = {
"ruler_name": "r_slice_ex",
"src_ops_alias": ["Slice", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I_0:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"acu_inter_flow": [],
"param_map": {"slice": {
    'begin':["INTS", "CODE",
             "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), tensor['Constant:out0'])"],
    'size':["INTS", "CODE",
            "self.slice_ex_size(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                "tensor['Constant:out0'], tensor['Constant_1:out0'])"],
}},
"blob_map": {"slice": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [10, -1]}
ruler_list.append(r_slice_ex)

r_slice_2 = {
"ruler_name": 'r_slice_2',
"src_ops_alias": ["Slice", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"],
                   ["Constant_2:out0", "Slice:in3"],
                   ["Constant_3:out0", "Slice:in4"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I_0:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"acu_inter_flow": [],
"param_map": {"slice": {'begin': ["INTS", "CODE",
                                  "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                  "tensor['Constant:out0'], tensor['Constant_2:out0'])"],
                        'size': ["INTS", "CODE",
                                 "self.slice_ex_size(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                 "tensor['Constant:out0'], tensor['Constant_1:out0'], tensor['Constant_2:out0'])"]
                        }},
"blob_map": {"slice": {}},
"priority_tip": 0,
"pre_condition": r_slice_pre_cond(steps_tensor='Constant_3:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_slice_2)

r_slice_3 = {
"ruler_name": 'r_slice_3',
"src_ops_alias": ["Slice", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"],
                   ["Constant_2:out0", "Slice:in3"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I_0:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"acu_inter_flow": [],
"param_map": {"slice": {'begin': ["INTS", "CODE",
                                  "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                  "tensor['Constant:out0'], tensor['Constant_2:out0'])"],
                        'size': ["INTS", "CODE",
                                 "self.slice_ex_size(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                 "tensor['Constant:out0'], tensor['Constant_1:out0'], tensor['Constant_2:out0'])"]}},
"blob_map": {"slice": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_slice_3)

r_slice_to_stride_slice = {
"ruler_name": "r_slice_to_stride_slice",
"src_ops_alias": ["Slice", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"], ["Constant_2:out0", "Slice:in3"],
                   ["Constant_3:out0", "Slice:in4"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["stridedslice"],
"src_acu_in_tensor_map": [["I_0:out0", "stridedslice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "stridedslice:out0"]],
"acu_inter_flow": [],
"param_map": {"stridedslice": {'slice_begin':
                                   ["INTS", "CODE",
                                    "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                    "tensor['Constant:out0'], tensor['Constant_2:out0'])"],
                               'slice_end':
                                   ["INTS", "CODE", "self.stride_slice_end(node['Slice'], self.shape_pick("
                                                    "tensor['I_0:out0']), tensor['Constant_1:out0'], "
                                                    "tensor['Constant_2:out0'])"],
                               'slice_strides': ["INTS", "CODE", "self.stride_slice_strides(node['Slice'], "
                                                 "self.shape_pick(tensor['I_0:out0']), "
                                                 "tensor['Constant_3:out0'], tensor['Constant_2:out0'])"
                                                 ]}},
"blob_map": {"stridedslice": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_slice_to_stride_slice)

r_upsample_l7_to_resize = {
"ruler_name": "upsample_l7_to_resize",
"src_ops_alias": ["Upsample"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Upsample:in0"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                  {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                    "type": ['STRING', 'CODE', "self.attr_pick(node['Upsample'], 'mode')"],}
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 8]}
ruler_list.append(r_upsample_l7_to_resize)

r_upsample_l9_to_resize = {
"ruler_name": "upsample_l9_to_resize",
"src_ops_alias": ["Upsample", "Constant"],
"src_inter_flow": [["Constant:out0", "Upsample:in1"]],
"src_in_anchor": [["I:out0", "Upsample:in0"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                  {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                    "type":
                       ['STRING',
                        'CODE',
                        "'bilinear' "
                        "if self.attr_pick(node['Upsample'], 'mode') == 'linear' "
                        "else self.attr_pick(node['Upsample'], 'mode')"],}
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 11,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [9, -1]}
ruler_list.append(r_upsample_l9_to_resize)

r_upsample_l9_scale_to_resize = {
"ruler_name": "r_upsample_l9_scale_to_resize",
"src_ops_alias": ["Upsample", ],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Upsample:in0"], ["I_1:out0", "Upsample:in1"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                  {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                    "type":
                       ['STRING',
                        'CODE',
                        "'bilinear' "
                        "if self.attr_pick(node['Upsample'], 'mode') == 'linear' "
                        "else self.attr_pick(node['Upsample'], 'mode')"],}
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [9, -1]}
ruler_list.append(r_upsample_l9_scale_to_resize)

r_resize_10 = {
"ruler_name": "r_resize_10",
"src_ops_alias": ["Resize", "Constant"],
"src_inter_flow": [["Constant:out0", "Resize:in1"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'VALUE', False],
        "half_pixel": ['BOOL', 'VALUE', False],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_10_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [0, 10]
}
ruler_list.append(r_resize_10)

r_resize = {
"ruler_name": "r_resize",
"src_ops_alias": ["Resize", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Resize:in1"], ["Constant_1:out0", "Resize:in2"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize)

r_resize_size = {
"ruler_name": "r_resize_size",
"src_ops_alias": ["Resize", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Constant:out0", "Resize:in1"],
                   ["Constant_1:out0", "Resize:in2"], ["Constant_2:out0", "Resize:in3"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_size)

r_resize_i0_constant_frozentablei1= {
"ruler_name": "r_resize_i0_constant_frozentablei1",
"src_ops_alias": ["Resize", "Constant"],
"src_inter_flow": [["Constant:out0", "Resize:in1"],
                   ["Constant:out0", "Resize:in2"]],
"src_in_anchor": [["I:out0", "Resize:in0"],["I_1:out0", "Resize:in3"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_i0_constant_frozentablei1)

r_resize_i0_constant0_constant1= {
"ruler_name": "r_resize_i0_constant0_constant1",
"src_ops_alias": ["Resize", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Resize:in1"], ["Constant:out0", "Resize:in2"],
                   ["Constant_1:out0", "Resize:in3"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {"image_resize": {}},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_resize_i0_constant0_constant1)

r_resize_i_frozentablei1= {
"ruler_name": "r_resize_i_frozentablei1",
"src_ops_alias": ["Resize"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Resize:in0"],["I_1:out0", "Resize:in3"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_i_frozentablei1)

r_resize_dynamic = {
"ruler_name": "r_resize_size",
"src_ops_alias": ["Resize"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Resize:in0"], ["I_1:out0", "Resize:in1"],
                  ["I_2:out0", "Resize:in2"], ["I_3:out0", "Resize:in3"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_dynamic)

r_resize_i0_scales = {
"ruler_name": "r_resize_i0_scales",
"src_ops_alias": ["Resize", "Constant"],
"src_inter_flow": [["Constant:out0", "Resize:in2"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]
}
ruler_list.append(r_resize_i0_scales)

r_pad_g1 = {
"ruler_name": "pad_g1",
"src_ops_alias": ["Pad"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"param_map": {"pad":
                  {'padding_value': ['ORIGIN', 'CODE', "self.map_pad_value(node['Pad'])"],
                   'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                   'padding_const': ['ORIGIN', 'CODE', "self.attr_pick(node['Pad'], 'value')"],
                   }
              },
"blob_map": {"pad": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [2, 10]}
ruler_list.append(r_pad_g1)

r_pad_11 = {
"ruler_name": "r_pad_11",
"src_ops_alias": ["Pad", "Constant"],
"src_inter_flow": [["Constant:out0", "Pad:in1"]],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"param_map": {"pad":
                  {'padding_value': ['ORIGIN', 'PYFUNC', r_pad_value_map()],
                   'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                   }
              },
"blob_map": {"pad": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_pad_11)

r_pad_1 = {
"ruler_name": "r_pad_1",
"src_ops_alias": ["Pad", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Pad:in1"], ["Constant_1:out0", "Pad:in2"]],
"src_in_anchor": [["I_0:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I_0:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"acu_inter_flow": [],
"param_map": {"pad": {'padding_value': ['ORIGIN', 'PYFUNC', r_pad_value_map()],
                      'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                      'padding_const': ['INT', 'CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
                   }},
"blob_map": {"pad": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_pad_1)

r_space2depth = {
"ruler_name": "r_space2depth",
"src_ops_alias": ["SpaceToDepth"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "SpaceToDepth:in0"]],
"src_out_tensor": ["SpaceToDepth:out0"],
"acu_lys_alias": ["space2depth"],
"src_acu_in_tensor_map": [["I:out0", "space2depth:in0"]],
"src_acu_out_tensor_map": [["SpaceToDepth:out0", "space2depth:out0"]],
"param_map":
    {"space2depth":
         {"block_size":
["INTS",
 "CODE",
 "[self.attr_pick(node['SpaceToDepth'], 'blocksize'), self.attr_pick(node['SpaceToDepth'], 'blocksize')]"]
          }
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_space2depth)

r_depth2space = {
"ruler_name": "r_depth2space",
"src_ops_alias": ["DepthToSpace"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "DepthToSpace:in0"]],
"src_out_tensor": ["DepthToSpace:out0"],
"acu_lys_alias": ["depth2space"],
"src_acu_in_tensor_map": [["I:out0", "depth2space:in0"]],
"src_acu_out_tensor_map": [["DepthToSpace:out0", "depth2space:out0"]],
"param_map":
{"depth2space":
     {"block_size":
          ["INT",
           "CODE",
           "self.attr_pick(node['DepthToSpace'], 'blocksize')"
           ],
      "mode":
          ["STRING",
           "CODE",
           "self.attr_pick(node['DepthToSpace'], 'mode')"
           ]
      }
 },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_depth2space)

r_clip_1 = {
"ruler_name": "r_clip_1",
"src_ops_alias": ["Clip"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE',
                                      "self.attr_pick(node['Clip'], 'max', 1.0)"],
                  'clip_value_min': ['FLOAT', 'CODE',
                                        "self.attr_pick(node['Clip'], 'min', -1.0)"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_clip_1)

r_clip_6 = {
"ruler_name": "r_clip_6",
"src_ops_alias": ["Clip"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE',
                                      "self.attr_pick(node['Clip'], 'max', np.inf)"],
                  'clip_value_min': ['FLOAT', 'CODE',
                                        "self.attr_pick(node['Clip'], 'min', -np.inf)"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, 10]}
ruler_list.append(r_clip_6)

r_clip_11 = {
"ruler_name": "r_clip_11",
"src_ops_alias": ["Clip", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Clip:in1"], ["Constant_1:out0", "Clip:in2"]],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE',
                                      "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
                  'clip_value_min': ['FLOAT', 'CODE',
                                        "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_clip_11)

r_clip_min = {
"ruler_name": "r_clip_min",
"src_ops_alias": ["Clip", "Constant"],
"src_inter_flow": [["Constant:out0", "Clip:in1"]],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE', "np.inf"],
                   'clip_value_min': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_clip_min)

r_clip_max = {
"ruler_name": "r_clip_max",
"src_ops_alias": ["Clip", "Constant"],
"src_inter_flow": [["Constant:out0", "Clip:in2"]],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
                   'clip_value_min': ['FLOAT', 'CODE', "-np.inf"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_clip_max)

r_clip_3inputs = {
"ruler_name": "r_clip_3inputs",
"src_ops_alias": ["Clip"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Clip:in0"], ["I_1:out0", "Clip:in1"], ["I_2:out0", "Clip:in2"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {"clipbyvalue": {'clip_value_max': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['I_2:out0'])"],
                              'clip_value_min': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['I_1:out0'])"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_clip_3inputs)

r_reducemean = {
"ruler_name": "r_reducemean",
"src_ops_alias": ["ReduceMean"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"]],
"src_out_tensor": ["ReduceMean:out0"],
"acu_lys_alias": ["reducemean"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemean:in0"]],
"src_acu_out_tensor_map": [["ReduceMean:out0", "reducemean:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemean':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceMean'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMean'], 'keepdims', 1)"]}},
"blob_map": {"reducemean": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducemean)

r_reducemax = {
"ruler_name": "r_reducemax",
"src_ops_alias": ["ReduceMax"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceMax:in0"]],
"src_out_tensor": ["ReduceMax:out0"],
"acu_lys_alias": ["reducemax"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemax:in0"]],
"src_acu_out_tensor_map": [["ReduceMax:out0", "reducemax:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemax':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceMax'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMax'], 'keepdims', 1)"]}},
"blob_map": {"reducemax": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducemax)

r_reducemin = {
"ruler_name": "r_reducemin",
"src_ops_alias": ["ReduceMin"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceMin:in0"]],
"src_out_tensor": ["ReduceMin:out0"],
"acu_lys_alias": ["reducemin"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemin:in0"]],
"src_acu_out_tensor_map": [["ReduceMin:out0", "reducemin:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemin':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceMin'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMin'], 'keepdims', 1)"]}},
"blob_map": {"reducemin": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducemin)

r_reducesum = {
"ruler_name": "r_reducesum",
"src_ops_alias": ["ReduceSum"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceSum:in0"]],
"src_out_tensor": ["ReduceSum:out0"],
"acu_lys_alias": ["reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceSum:out0", "reducesum:out0"]],
"acu_inter_flow": [],
"param_map": {'reducesum':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceSum'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSum'], 'keepdims', 1)"]}},
"blob_map": {"reducesum": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducesum)

r_reducesum_with_constant_axes = {
"ruler_name": "r_reducesum_with_constant_axes",
"src_ops_alias": ["ReduceSum", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceSum:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceSum:in0"]],
"src_out_tensor": ["ReduceSum:out0"],
"acu_lys_alias": ["reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceSum:out0", "reducesum:out0"]],
"acu_inter_flow": [],
"param_map":
{'reducesum':
     {'axis_list':
          ["INTS",
           "CODE",
           "self.reducesum_constant_axis_list(node['ReduceSum'], tensor['Constant:out0'], "
           "self.shape_pick(tensor['I_0:out0']))"],
      'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSum'], 'keepdims', 1)"]
      }
 },
"blob_map": {"reducesum": {}},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceSum'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_reducesum_with_constant_axes)

r_reducesum_to_noop_with_constant_axes = {
"ruler_name": "r_reducesum_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceSum", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceSum:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceSum:in0"]],
"src_out_tensor": ["ReduceSum:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceSum:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map":{},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceSum'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_reducesum_to_noop_with_constant_axes)

r_reduceprod = {
"ruler_name": "r_reduceprod",
"src_ops_alias": ["ReduceProd"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceProd:in0"]],
"src_out_tensor": ["ReduceProd:out0"],
"acu_lys_alias": ["reduceprod"],
"src_acu_in_tensor_map": [["I_0:out0", "reduceprod:in0"]],
"src_acu_out_tensor_map": [["ReduceProd:out0", "reduceprod:out0"]],
"acu_inter_flow": [],
"param_map":
{'reduceprod':
     {'axis_list':
          ["INTS",
           "CODE",
           "self.reducex_axis_list(node['ReduceProd'], self.shape_pick(tensor['I_0:out0']))"],
      'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceProd'], 'keepdims', 1)"]
      }
 },
"blob_map": {"reduceprod": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reduceprod)

r_reducel1 = {
"ruler_name": "r_reducel1",
"src_ops_alias": ["ReduceL1"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceL1:in0"]],
"src_out_tensor": ["ReduceL1:out0"],
"acu_lys_alias": ["abs", "reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "abs:in0"]],
"src_acu_out_tensor_map": [["ReduceL1:out0", "reducesum:out0"]],
"acu_inter_flow": [["abs:out0", "reducesum:in0"]],
"param_map": {"abs": {},
              "reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceL1'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['ReduceL1'], 'keepdims', 1)"]}},
"blob_map": {"abs": {},
             "reducesum": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reducel1)

r_reducel2 = {
"ruler_name": 'r_reducel2',
"src_ops_alias": ["ReduceL2"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceL2:in0"]],
"src_out_tensor": ["ReduceL2:out0"],
"acu_lys_alias": ["reducesum", "multiply", "sqrt"],
"src_acu_in_tensor_map": [["I_0:out0", "multiply:in0"], ["I_0:out0", "multiply:in1"]],
"src_acu_out_tensor_map": [["ReduceL2:out0", "sqrt:out0"]],
"acu_inter_flow": [["multiply:out0", "reducesum:in0"], ["reducesum:out0", "sqrt:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceL2'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceL2'], 'keepdims', 1)"]},
              "multiply":{},
              "sqrt":{}},
"blob_map": {"reducesum": {},
             "multiply": {},
             "sqrt":{}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reducel2)

r_reducelogsum = {
"ruler_name": "r_reducelogsum",
"src_ops_alias": ["ReduceLogSum"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceLogSum:in0"]],
"src_out_tensor": ["ReduceLogSum:out0"],
"acu_lys_alias": ["reducesum", "log"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSum:out0", "log:out0"]],
"acu_inter_flow": [["reducesum:out0", "log:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceLogSum'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceLogSum'], 'keepdims', 1)"]},
              "log": {}},
"blob_map": {"reducesum": {},
             "log": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reducelogsum)

r_reducelogsumexp = {
"ruler_name": "r_reducelogsumexp",
"src_ops_alias": ["ReduceLogSumExp"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceLogSumExp:in0"]],
"src_out_tensor": ["ReduceLogSumExp:out0"],
"acu_lys_alias": ["exp", "reducesum", "log"],
"src_acu_in_tensor_map": [["I_0:out0", "exp:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSumExp:out0", "log:out0"]],
"acu_inter_flow": [["exp:out0", "reducesum:in0"], ["reducesum:out0", "log:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceLogSumExp'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceLogSumExp'], 'keepdims', 1)"]},
              "exp": {},
              "log": {}},
"blob_map": {"reducesum": {},
             "exp": {},
             "log": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducelogsumexp)

r_ReduceSumSquare = {
"ruler_name": "r_ReduceSumSquare",
"src_ops_alias": ["ReduceSumSquare"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceSumSquare:in0"]],
"src_out_tensor": ["ReduceSumSquare:out0"],
"acu_lys_alias": ["multiply", "reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "multiply:in0"], ["I_0:out0", "multiply:in1"]],
"src_acu_out_tensor_map": [["ReduceSumSquare:out0", "reducesum:out0"]],
"acu_inter_flow": [["multiply:out0", "reducesum:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceSumSquare'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSumSquare'], 'keepdims', 1)"]},
              "multiply": {}},
"blob_map": {"reducesum": {},
             "multiply": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_ReduceSumSquare)

r_gather = {
"ruler_name": "r_gather",
"src_ops_alias": ["Gather"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Gather:in0"], ["I_1:out0", "Gather:in1"]],
"src_out_tensor": ["Gather:out0"],
"acu_lys_alias": ["gather"],
"src_acu_in_tensor_map": [["I:out0", "gather:in0"], ["I_1:out0", "gather:in1"]],
"src_acu_out_tensor_map": [["Gather:out0", "gather:out0"]],
"acu_inter_flow": [],
"param_map": {"gather": {'axis': ['INT', 'CODE', "self.attr_pick(node['Gather'], 'axis', 0)"]}},
"blob_map": {"gather": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather)

r_gather_nd = {
"ruler_name": "r_gather_nd",
"src_ops_alias": ["GatherND"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GatherND:in0"], ["I_1:out0", "GatherND:in1"]],
"src_out_tensor": ["GatherND:out0"],
"acu_lys_alias": ["gathernd"],
"src_acu_in_tensor_map": [["I:out0", "gathernd:in0"], ["I_1:out0", "gathernd:in1"]],
"src_acu_out_tensor_map": [["GatherND:out0", "gathernd:out0"]],
"acu_inter_flow": [],
"param_map": {"gathernd": {'batch_dims': ['INT', 'CODE', "self.attr_pick(node['GatherND'], 'batch_dims', 0)"]}},
"blob_map": {"gathernd": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather_nd)

r_softplus = {
"ruler_name": 'r_softplus',
"src_ops_alias": ["Softplus"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Softplus:in0"]],
"src_out_tensor": ["Softplus:out0"],
"acu_lys_alias": ["softrelu"],
"src_acu_in_tensor_map": [["I_0:out0", "softrelu:in0"]],
"src_acu_out_tensor_map": [["Softplus:out0", "softrelu:out0"]],
"acu_inter_flow": [],
"param_map": {"softrelu": {}},
"blob_map": {"softrelu": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_softplus)

r_tile = {
"ruler_name": "r_tile",
"src_ops_alias": ["Tile", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Tile:in1"]],
"src_in_anchor": [["I:out0", "Tile:in0"]],
"src_out_tensor": ["Tile:out0"],
"acu_lys_alias": ["tile"],
"src_acu_in_tensor_map": [["I:out0", "tile:in0"]],
"src_acu_out_tensor_map": [["Tile:out0", "tile:out0"]],
"param_map":
    {"tile":
         {"multiples": ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_tile)

r_argmin = {
"ruler_name": "r_argmin",
"src_ops_alias": ["ArgMin"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ArgMin:in0"]],
"src_out_tensor": ["ArgMin:out0"],
"acu_lys_alias": ["argmin"],
"src_acu_in_tensor_map":[["I:out0", "argmin:in0"]],
"src_acu_out_tensor_map": [["ArgMin:out0", "argmin:out0"]],
"param_map":{
  "argmin":
    {
      'axis': ['INT', 'CODE', "self.attr_pick(node['ArgMin'], 'axis', 0)"],
      'keepdims': ['BOOL', 'CODE', "self.attr_pick(node['ArgMin'], 'keepdims', 1)"],
      'select_last_index': ['BOOL', 'CODE', "self.attr_pick(node['ArgMin'], 'select_last_index', 0)"],
      'output_type': ['STRING', 'CODE', "self.dtype_pick(tensor['ArgMin:out0'])"]
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_argmin)

r_argmax = {
"ruler_name": "r_argmax",
"src_ops_alias": ["ArgMax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ArgMax:in0"]],
"src_out_tensor": ["ArgMax:out0"],
"acu_lys_alias": ["argmax"],
"src_acu_in_tensor_map":[["I:out0", "argmax:in0"]],
"src_acu_out_tensor_map": [["ArgMax:out0", "argmax:out0"]],
"param_map":{
  "argmax": {'output_type': ['STRING', 'CODE', "self.dtype_pick(tensor['ArgMax:out0'])"]}
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_argmax)

r_neg = {
"ruler_name": "r_neg",
"src_ops_alias": ["Neg"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Neg:in0"]],
"src_out_tensor": ["Neg:out0"],
"acu_lys_alias": ["neg"],
"src_acu_in_tensor_map": [["I:out0", "neg:in0"]],
"src_acu_out_tensor_map": [["Neg:out0", "neg:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_neg)

r_sin = {
"ruler_name": "r_sin",
"src_ops_alias": ["Sin"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Sin:in0"]],
"src_out_tensor": ["Sin:out0"],
"acu_lys_alias": ["sin"],
"src_acu_in_tensor_map": [["I_0:out0", "sin:in0"]],
"src_acu_out_tensor_map": [["Sin:out0", "sin:out0"]],
"acu_inter_flow": [],
"param_map": {"sin": {}},
"blob_map": {"sin": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_sin)

r_reverse_sequence = {
"ruler_name": "r_reverse_sequence",
"src_ops_alias": ["ReverseSequence"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ReverseSequence:in0"], ['I_0:out0', "ReverseSequence:in1"]],
"src_out_tensor": ["ReverseSequence:out0"],
"acu_lys_alias": ["reverse_sequence"],
"src_acu_in_tensor_map":[["I:out0", "reverse_sequence:in0"], ['I_0:out0', "reverse_sequence:in1"]],
"src_acu_out_tensor_map": [["ReverseSequence:out0", "reverse_sequence:out0"]],
"param_map": {
    "reverse_sequence":{
        "seq_axis":["INT", "CODE", "self.attr_pick(node['ReverseSequence'], 'time_axis', 0)"],
        "batch_axis":["INT", "CODE", "self.attr_pick(node['ReverseSequence'], 'batch_axis', 1)"],
    },
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reverse_sequence)

r_where = {
"ruler_name": "r_where",
"src_ops_alias": ["Where"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Where:in0"], ["I_1:out0", "Where:in1"], ["I_2:out0", "Where:in2"]],
"src_out_tensor": ["Where:out0"],
"acu_lys_alias": ["where"],
"src_acu_in_tensor_map": [["I_0:out0", "where:in0"], ["I_1:out0", "where:in1"], ["I_2:out0", "where:in2"]],
"src_acu_out_tensor_map": [["Where:out0", "where:out0"]],
"acu_inter_flow": [],
"param_map": {"where": {}},
"blob_map": {"where": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_where)

r_matmul = {
"ruler_name": "r_matmul",
"src_ops_alias": ["MatMul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MatMul:in0"], ['I_1:out0', "MatMul:in1"]],
"src_out_tensor": ["MatMul:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map":[["I:out0", "matmul:in0"], ['I_1:out0', "matmul:in1"]],
"src_acu_out_tensor_map": [["MatMul:out0", "matmul:out0"]],
"param_map": {"matmul": {'transpose_a': ['BOOL', 'CODE', "self.attr_pick(node['MatMul'], 'transpose_a', False)"],
                         'transpose_b': ['BOOL', 'CODE', "self.attr_pick(node['MatMul'], 'transpose_b', False)"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_matmul)

r_xor = {
"ruler_name": "r_logical_xor",
"src_ops_alias": ["Xor"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Xor:in0"], ["I_1:out0", "Xor:in1"]],
"src_out_tensor": ["Xor:out0"],
"acu_lys_alias": ["not_equal"],
"src_acu_in_tensor_map": [["I_0:out0", "not_equal:in0"], ["I_1:out0", "not_equal:in1"]],
"src_acu_out_tensor_map": [["Xor:out0", "not_equal:out0"]],
"acu_inter_flow": [],
"param_map": {"not_equal": {}},
"blob_map": {"not_equal": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_xor)

r_expand = {
"ruler_name": "r_expand",
"src_ops_alias": ["Expand", "Constant"],
"src_inter_flow": [["Constant:out0", "Expand:in1"]],
"src_in_anchor": [["I_0:out0", "Expand:in0"]],
"src_out_tensor": ["Expand:out0"],
"acu_lys_alias": ["expand_broadcast"],
"src_acu_in_tensor_map": [["I_0:out0", "expand_broadcast:in0"]],
"src_acu_out_tensor_map": [["Expand:out0", "expand_broadcast:out0"]],
"acu_inter_flow": [],
"param_map": {
    "expand_broadcast": {
        "shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]
    }
},
"blob_map": {"expand_broadcast": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_expand)

r_expand_dynamic = {
"ruler_name": "r_expand_dynamic",
"src_ops_alias": ["Expand"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Expand:in0"],["I_1:out0", "Expand:in1"]],
"src_out_tensor": ["Expand:out0"],
"acu_lys_alias": ["expand_broadcast"],
"src_acu_in_tensor_map": [["I_0:out0", "expand_broadcast:in0"]],
"src_acu_out_tensor_map": [["Expand:out0", "expand_broadcast:out0"]],
"acu_inter_flow": [],
"param_map": {
    "expand_broadcast": {
        "shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['I_1:out0'])"]
    }
},
"blob_map": {"expand_broadcast": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_expand_dynamic)

r_quantizelinear = {
"ruler_name": "r_quantizelinear",
"src_ops_alias": ["QuantizeLinear", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "QuantizeLinear:in1"], ["Constant_1:out0", "QuantizeLinear:in2"]],
"src_in_anchor": [["I_0:out0", "QuantizeLinear:in0"]],
"src_out_tensor": ["QuantizeLinear:out0"],
"acu_lys_alias": ["quantize"],
"src_acu_in_tensor_map": [["I_0:out0", "quantize:in0"]],
"src_acu_out_tensor_map": [["QuantizeLinear:out0", "quantize:out0"]],
"acu_inter_flow": [],
"param_map": {"quantize": {}},
"blob_map": {"quantize": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_quantizelinear)
#QuantizeLinear:QuantizeLinear_Y;Constant:Initializer_X_SCALE;Constant_1:Initializer_X_ZP

r_dequantizelinear = {
"ruler_name": "r_dequantizelinear",
"src_ops_alias": ["DequantizeLinear", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "DequantizeLinear:in1"], ["Constant_1:out0", "DequantizeLinear:in2"]],
"src_in_anchor": [["I_0:out0", "DequantizeLinear:in0"]],
"src_out_tensor": ["DequantizeLinear:out0"],
"acu_lys_alias": ["dequantize"],
"src_acu_in_tensor_map": [["I_0:out0", "dequantize:in0"]],
"src_acu_out_tensor_map": [["DequantizeLinear:out0", "dequantize:out0"]],
"acu_inter_flow": [],
"param_map": {"dequantize": {}},
"blob_map": {"dequantize": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_dequantizelinear)
#DequantizeLinear:DequantizeLinear_Y;Constant:Initializer_scale;Constant_1:Initializer_zp

# r_qlinearmatmul = {
# "ruler_name": "qlinearmatmul",
# "src_ops_alias": ["QLinearMatMul", "Constant", "Constant_1", "Constant_2", "Constant_3", "Constant_4", "Constant_5"],
# "src_inter_flow": [["Constant:out0", "QLinearMatMul:in1"], ["Constant_1:out0", "QLinearMatMul:in2"],
#                    ["Constant_2:out0", "QLinearMatMul:in4"], ["Constant_3:out0", "QLinearMatMul:in5"],
#                    ["Constant_4:out0", "QLinearMatMul:in6"], ["Constant_5:out0", "QLinearMatMul:in7"]],
# "src_in_anchor": [["I_0:out0", "QLinearMatMul:in0"], ["I_1:out0", "QLinearMatMul:in3"]],
# "src_out_tensor": ["QLinearMatMul:out0"],
# "acu_lys_alias": ["matmul"],
# "src_acu_in_tensor_map": [["I_0:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
# "src_acu_out_tensor_map": [["QLinearMatMul:out0", "matmul:out0"]],
# "acu_inter_flow": [],
# "param_map": {"matmul": {}},
# "blob_map": {"matmul": {}},
# "priority_tip": 0,
# "pre_condition": None,
# "src_ops_main_version": None,
# "src_ops_minior_version": [1, -1]}
# ruler_list.append(r_qlinearmatmul)

r_qlinearmatmul = {
"ruler_name": "qlinearmatmul",
"src_ops_alias": ["QLinearMatMul", "Constant", "Constant_1", "Constant_2", "Constant_3", "Constant_4", "Constant_5"],
"src_inter_flow": [["Constant:out0", "QLinearMatMul:in1"], ["Constant_1:out0", "QLinearMatMul:in2"],
                   ["Constant_2:out0", "QLinearMatMul:in4"], ["Constant_3:out0", "QLinearMatMul:in5"],
                   ["Constant_4:out0", "QLinearMatMul:in6"], ["Constant_5:out0", "QLinearMatMul:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearMatMul:in0"], ["I_1:out0", "QLinearMatMul:in3"]],
"src_out_tensor": ["QLinearMatMul:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map": [["I_0:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
"src_acu_out_tensor_map": [["QLinearMatMul:out0", "matmul:out0"]],
"acu_inter_flow": [],
"param_map": {"matmul": {}},
"blob_map": {"matmul": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearmatmul)
#QLinearMatMul:QLinearMatMul_Y;Constant:Initializer_X0_SCALE;Constant_1:Initializer_X0_ZP;
#Constant_2:Initializer_X1_SCALE;Constant_3:Initializer_X1_ZP;Constant_4:Initializer_Y_SCALE;
#Constant_5:Initializer_Y_ZP

r_qlinearconv_no_bias = {
"ruler_name": "qlinearconv_no_bias",
"src_ops_alias": ["QLinearConv", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5", "Constant_6"],
"src_inter_flow": [["Constant:out0", "QLinearConv:in1"], ["Constant_1:out0", "QLinearConv:in2"],
                   ["Constant_2:out0", "QLinearConv:in3"], ["Constant_3:out0", "QLinearConv:in4"],
                   ["Constant_4:out0", "QLinearConv:in5"], ["Constant_5:out0", "QLinearConv:in6"],
                   ["Constant_6:out0", "QLinearConv:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearConv:in0"]],
"src_out_tensor": ["QLinearConv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I_0:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["QLinearConv:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[0]"],
# "pad_method":
#    ["STRING", "CODE", "'auto' if self.attr_pick(node['QLinearConv'], 'pads', None) == None else 'padding_const'"],
"pad_method":
   ["STRING", "CODE",
    "'auto' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') != 'NOTSET' else 'padding_const'"],
"bias": ["BOOL", "VALUE", False],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default =[1,1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default=[1,1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['QLinearConv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['QLinearConv'], 'dilations'), int)"\
      " else self.attr_pick(node['QLinearConv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] "
    "else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['QLinearConv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_2:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "self.tensor_to_numpy(tensor['Constant_4:out0']))"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearconv_no_bias)
#QLinearConv:QLinearConv_Y;Constant:Initializer_X_SCALE;Constant_1:Initializer_X_ZP;Constant_2:Initializer_W;
#Constant_3:Initializer_W_SCALE;Constant_4:Initializer_W_ZP;Constant_5:Initializer_Y_SCALE;Constant_6:Initializer_Y_ZP

r_qlinearconv_with_bias = {
"ruler_name": "qlinearconv_with_bias",
"src_ops_alias": ["QLinearConv", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5", "Constant_6", "Constant_7"],
"src_inter_flow": [["Constant:out0", "QLinearConv:in1"], ["Constant_1:out0", "QLinearConv:in2"],
                   ["Constant_2:out0", "QLinearConv:in3"], ["Constant_3:out0", "QLinearConv:in4"],
                   ["Constant_4:out0", "QLinearConv:in5"], ["Constant_5:out0", "QLinearConv:in6"],
                   ["Constant_6:out0", "QLinearConv:in7"], ["Constant_7:out0", "QLinearConv:in8"]],
"src_in_anchor": [["I_0:out0", "QLinearConv:in0"]],
"src_out_tensor": ["QLinearConv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I_0:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["QLinearConv:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[0]"],
"pad_method":
   ["STRING", "CODE",
    "'auto' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') != 'NOTSET' else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default =[1,1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default=[1,1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['QLinearConv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['QLinearConv'], 'dilations'), int)"\
      " else self.attr_pick(node['QLinearConv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] "
    "else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['QLinearConv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_2:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "self.tensor_to_numpy(tensor['Constant_4:out0']))"],
                  "bias": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_7:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant:out0'])"
                                     "*self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "np.array([0]*self.shape_pick(tensor['Constant_2:out0'])[0],dtype=np.int32))"]
                  }
             },
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearconv_with_bias)
#QLinearConv:QLinearConv_Y;Constant:Initializer_X_SCALE;Constant_1:Initializer_X_ZP;Constant_2:Initializer_W;
#Constant_3:Initializer_W_SCALE;Constant_4:Initializer_W_ZP;Constant_5:Initializer_Y_SCALE;
#Constant_6:Initializer_Y_ZP;Constant_7:Initializer_B

r_qlinearconv_1d_with_bias_share_zp = {
"ruler_name": "qlinearconv_1d_with_bias_share_zp",
"src_ops_alias": ["QLinearConv", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5"],
"src_inter_flow": [["Constant:out0", "QLinearConv:in1"], ["Constant_1:out0", "QLinearConv:in2"],
                   ["Constant_2:out0", "QLinearConv:in3"], ["Constant_3:out0", "QLinearConv:in4"],
                   ["Constant_1:out0", "QLinearConv:in5"], ["Constant_4:out0", "QLinearConv:in6"],
                   ["Constant_1:out0", "QLinearConv:in7"], ["Constant_5:out0", "QLinearConv:in8"]],
"src_in_anchor": [["I_0:out0", "QLinearConv:in0"]],
"src_out_tensor": ["QLinearConv:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I_0:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["QLinearConv:out0", "conv1d:out0"]],
"acu_inter_flow": [],
"param_map":
{"conv1d":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[0]"],
"pad_method":
   ["STRING", "CODE",
    "'auto' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') != 'NOTSET' else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"group_number": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'group', 1)"],
"ksize": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[2]"],
"stride": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default=[1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['QLinearConv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['QLinearConv'], 'dilations'), int)"\
      " else self.attr_pick(node['QLinearConv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] "
    "else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['QLinearConv'], 'pads', [ 0, 0,]), [0, 1])]"]
}
},
"blob_map": {"conv1d":
                 {"weight": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_2:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "self.tensor_to_numpy(tensor['Constant_1:out0']))"],
                  "bias": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_5:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant:out0'])"
                                     "*self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "np.array([0]*self.shape_pick(tensor['Constant_2:out0'])[0],dtype=np.int32))"]
                  }
             },
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearconv_1d_with_bias_share_zp)

r_hard_swish = {
"ruler_name": "r_hard_swish",
"src_ops_alias": ["Div", "Mul", "Constant", "Clip", "Add", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Mul:out0", "Div:in0"], ["Constant:out0", "Div:in1"], ["Clip:out0", "Mul:in1"],
                   ["Add:out0", "Clip:in0"], ["Constant_1:out0", "Clip:in1"], ["Constant_2:out0", "Clip:in2"],
                   ["Constant_3:out0", "Add:in1"]],
"src_in_anchor": [["I_0:out0", "Add:in0"], ["I_0:out0", "Mul:in0"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["hard_swish"],
"src_acu_in_tensor_map": [["I_0:out0", "hard_swish:in0"]],
"src_acu_out_tensor_map": [["Div:out0", "hard_swish:out0"]],
"acu_inter_flow": [],
"param_map": {"hard_swish": {}},
"blob_map": {"hard_swish": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_hard_swish)

r_nonzero = {
"ruler_name": "r_nonzero",
"src_ops_alias": ["NonZero"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "NonZero:in0"]],
"src_out_tensor": ["NonZero:out0"],
"acu_lys_alias": ["nonzero"],
"src_acu_in_tensor_map": [["I:out0", "nonzero:in0"]],
"src_acu_out_tensor_map": [["NonZero:out0", "nonzero:out0"]],
"param_map": {"nonzero": {'output_type': ['STRING', 'CODE', "self.dtype_pick(tensor['NonZero:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_nonzero)

r_shape = {
"ruler_name": "r_shape",
"src_ops_alias": ["Shape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Shape:in0"]],
"src_out_tensor": ["Shape:out0"],
"acu_lys_alias": ["shapelayer"],
"src_acu_in_tensor_map": [["I:out0", "shapelayer:in0"]],
"src_acu_out_tensor_map": [["Shape:out0", "shapelayer:out0"]],
"param_map": {"shapelayer": {"out_type": ["STRING", "CODE", "self.dtype_pick(tensor['Shape:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_shape)

r_sign = {
"ruler_name": "r_sign",
"src_ops_alias": ["Sign"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sign:in0"]],
"src_out_tensor": ["Sign:out0"],
"acu_lys_alias": ["sign"],
"src_acu_in_tensor_map": [["I:out0", "sign:in0"]],
"src_acu_out_tensor_map": [["Sign:out0", "sign:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_sign)

r_size = {
"ruler_name": "r_size",
"src_ops_alias": ["Size"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Size:in0"]],
"src_out_tensor": ["Size:out0"],
"acu_lys_alias": ["size"],
"src_acu_in_tensor_map": [["I:out0", "size:in0"]],
"src_acu_out_tensor_map": [["Size:out0", "size:out0"]],
"param_map": {"size": {"out_type": ["STRING", "CODE", "self.dtype_pick(tensor['Size:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_size)

r_scatter_nd = {
"ruler_name": "r_scatter_nd",
"src_ops_alias": ["ScatterND"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ScatterND:in0"], ["I_1:out0", "ScatterND:in1"], ["I_2:out0", "ScatterND:in2"]],
"src_out_tensor": ["ScatterND:out0"],
"acu_lys_alias": ["scatter_nd_update"],
"src_acu_in_tensor_map": [["I:out0", "scatter_nd_update:in0"], ["I_1:out0", "scatter_nd_update:in1"],
                          ["I_2:out0", "scatter_nd_update:in2"]],
"src_acu_out_tensor_map": [["ScatterND:out0", "scatter_nd_update:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_scatter_nd)

r_mean_variance_normalization = {
"ruler_name": "r_mean_variance_normalization",
"src_ops_alias": ["MeanVarianceNormalization"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MeanVarianceNormalization:in0"]],
"src_out_tensor": ["MeanVarianceNormalization:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["MeanVarianceNormalization:out0", "instancenormalize:out0"]],
"param_map":{
    "instancenormalize":{
        'axis':['INTS', 'CODE', "[p for p in self.attr_pick(node['MeanVarianceNormalization'], 'axes', [0, 2, 3])]"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mean_variance_normalization)

def gen_onnx_ruler(dst_path):
    # print(json.dumps(ruler_list))
    dst_path = os.path.join(dst_path, 'onnx_ruler_db.json')

    with open(dst_path, 'w+') as f:
        json.dump(ruler_list, f, indent=1)

    # To Verify ruler follow synatx
    with open(dst_path, 'r') as f:
        x_val = json.load(f)
def main():
    gen_onnx_ruler(sys.argv[1])

if  __name__ == '__main__':
    main()

