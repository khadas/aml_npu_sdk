import json
import sys
import os

import dill

#support build-in functions:
'''
IR List
Reference Document @ pytorch.github.io/docs/master/jit.html#builtin-functions
@pytorch/torch/csrc/jit/autodiff.cpp
"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
"aten::t(Tensor self) -> Tensor",
"aten::dropout(Tensor input, float p, bool train) -> Tensor",
"aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training,
 float momentum, float eps, bool cudnn_enabled) -> Tensor",
"aten::conv1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups)
 -> Tensor",
"aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups)
 -> Tensor",
"aten::conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups)
 -> Tensor",
"aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad) -> Tensor",
"aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding,
 int groups, int[] dilation) -> Tensor",
"aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding,
 int groups, int[] dilation) -> Tensor",
"aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding,
 int groups, int[] dilation) -> Tensor",
"aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation,
 bool transposed, int[] output_padding, int groups) -> Tensor",
"aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation,
 bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
"aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor",
"aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor",
"aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor",
"aten::avg_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad)
 -> Tensor",
"aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad)
 -> Tensor",
"aten::avg_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad)
 -> Tensor",
"aten::max_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode)
 -> Tensor",
"aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode)
 -> Tensor",
"aten::max_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode)
 -> Tensor",
"aten::max_unpool2d(Tensor self, Tensor indices, int[] output_size) -> Tensor",
"aten::max_unpool3d(Tensor self, Tensor indices, int[] output_size, int[] stride, int[] padding) -> Tensor",
"aten::reflection_pad1d(Tensor self, int[] padding) -> Tensor",
"aten::reflection_pad2d(Tensor self, int[] padding) -> Tensor",
"aten::replication_pad1d(Tensor self, int[] padding) -> Tensor",
"aten::replication_pad2d(Tensor self, int[] padding) -> Tensor",
"aten::replication_pad3d(Tensor self, int[] padding) -> Tensor",
"aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners) -> Tensor",
"aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners) -> Tensor",
"aten::upsample_nearest1d(Tensor self, int[] output_size) -> Tensor",
"aten::upsample_nearest2d(Tensor self, int[] output_size) -> Tensor",
"aten::upsample_nearest3d(Tensor self, int[] output_size) -> Tensor",
"aten::upsample_trilinear3d(Tensor self, int[] output_size, bool align_corners) -> Tensor",
"aten::prelu(Tensor self, Tensor weight) -> Tensor",
"aten::view(Tensor self, int[] size) -> Tensor"
"aten::max_pool2d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation,
bool ceil_mode) -> (Tensor, Tensor)"
"aten::max_pool3d_with_indices(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation,
bool ceil_mode) -> (Tensor, Tensor)",
 "aten::abs(Tensor self) -> Tensor",
"aten::acos(Tensor self) -> Tensor",
"aten::neg(Tensor self) -> Tensor",
"aten::t(Tensor self) -> Tensor",
"aten::sigmoid(Tensor self) -> Tensor",
"aten::tanh(Tensor self) -> Tensor",
"aten::relu(Tensor self) -> Tensor",
"aten::asin(Tensor self) -> Tensor",
"aten::atan(Tensor self) -> Tensor",
"aten::ceil(Tensor self) -> Tensor",
"aten::clone(Tensor self) -> Tensor",
"aten::contiguous(Tensor self) -> Tensor",
"aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
"aten::celu(Tensor self, Scalar alpha) -> Tensor",
"aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
"aten::clamp_max(Tensor self, Scalar max) -> Tensor",
"aten::clamp_min(Tensor self, Scalar min) -> Tensor",
"aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
"aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
"aten::cos(Tensor self) -> Tensor",
"aten::cosh(Tensor self) -> Tensor",
"aten::digamma(Tensor self) -> Tensor",
"aten::dropout(Tensor input, float p, bool train) -> Tensor",
"aten::elu(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor",
"aten::erf(Tensor self) -> Tensor",
"aten::erfc(Tensor self) -> Tensor",
"aten::erfinv(Tensor self) -> Tensor",
"aten::exp(Tensor self) -> Tensor",
"aten::expm1(Tensor self) -> Tensor",
"aten::log(Tensor self) -> Tensor",
"aten::log10(Tensor self) -> Tensor",
"aten::log1p(Tensor self) -> Tensor",
"aten::log2(Tensor self) -> Tensor",
"aten::log_sigmoid(Tensor self) -> Tensor",
"aten::log_softmax(Tensor self, int dim) -> Tensor",
"aten::floor(Tensor self) -> Tensor",
"aten::frac(Tensor self) -> Tensor",
"aten::flip(Tensor self, int[] dims) -> Tensor",
"aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
"aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
"aten::hardshrink(Tensor self, Scalar lambd) -> Tensor",
"aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
"aten::glu(Tensor self, int dim) -> Tensor",
"aten::inverse(Tensor self) -> Tensor",
"aten::leaky_relu(Tensor self, Scalar negative_slope) -> Tensor",
"aten::lgamma(Tensor self) -> Tensor",
"aten::mvlgamma(Tensor self, int p) -> Tensor",
"aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
"aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
"aten::permute(Tensor self, int[] dims) -> Tensor",
"aten::pin_memory(Tensor self) -> Tensor",
"aten::pinverse(Tensor self, float rcond) -> Tensor",
"aten::reciprocal(Tensor self) -> Tensor",
"aten::relu(Tensor self) -> Tensor",
"aten::round(Tensor self) -> Tensor",
"aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
"aten::rsqrt(Tensor self) -> Tensor",
"aten::selu(Tensor self) -> Tensor",
"aten::sigmoid(Tensor self) -> Tensor",
"aten::sign(Tensor self) -> Tensor",
"aten::sin(Tensor self) -> Tensor",
"aten::sinh(Tensor self) -> Tensor",
"aten::softmax(Tensor self, int dim) -> Tensor",
"aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
"aten::softshrink(Tensor self, Scalar lambd) -> Tensor",
"aten::sqrt(Tensor self) -> Tensor",
"aten::tan(Tensor self) -> Tensor",
"aten::tanh(Tensor self) -> Tensor",
"aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
"aten::transpose(Tensor self, int dim0, int dim1) -> Tensor",
"aten::tril(Tensor self, int diagonal) -> Tensor",
"aten::triu(Tensor self, int diagonal) -> Tensor",
"aten::trunc(Tensor self) -> Tensor",
"aten::rot90(Tensor self, int k, int[] dims) -> Tensor",
"aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
"aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor",
"aten::alias(Tensor self) -> Tensor",
"aten::cumprod(Tensor self, int dim) -> Tensor",
"aten::cumsum(Tensor self, int dim) -> Tensor",
"aten::empty_like(Tensor self) -> Tensor",
"aten::full_like(Tensor self, Scalar fill_value) -> Tensor",
"aten::ones_like(Tensor self) -> Tensor",
"aten::rand_like(Tensor self) -> Tensor",
"aten::randint_like(Tensor self, int high) -> Tensor",
"aten::randint_like(Tensor self, int low, int high) -> Tensor",
"aten::randn_like(Tensor self) -> Tensor",
"aten::zeros_like(Tensor self) -> Tensor",
"aten::sum(Tensor self, int[] dim, bool keepdim) -> Tensor",
"aten::sum(Tensor self) -> Tensor",
"aten::squeeze(Tensor self, int dim) -> Tensor"

"aten::empty(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::full(int[] size, Scalar fill_value, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
"aten::reshape(Tensor self, int[] shape) -> Tensor"

"aten::pixel_shuffle(Tensor self, int upscale_factor) -> (Tensor) ;"

'''

'''
    def tensor_to_numpy(self, tensor, trans=None):
    def tensor_shape(self, tensor):
    def reshape_shape(self, orig_shape):
    def map_pad_value(self, tensor):
'''
ruler_list = list()

r_upsampe_as_resize = {
"ruler_name": "r_leakrelu",
"src_ops_alias": [r"aten::upsample_nearest2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::upsample_nearest2d:in0"]],
"src_out_tensor": [r"aten::upsample_nearest2d:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [[r"aten::upsample_nearest2d:out0", "image_resize:out0"]],
"param_map": {'image_resize':{'new_size':['ORIGIN', 'CODE',"self.attr_pick('aten::upsample_nearest2d', 'a_1')"],
                              'type': ['STRING', 'VALUE', 'nearest'],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_upsampe_as_resize)

r_leakrelu = {
"ruler_name": "r_leakrelu",
"src_ops_alias": [r"aten::leaky_relu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::leaky_relu:in0"]],
"src_out_tensor": [r"aten::leaky_relu:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [[r"aten::leaky_relu:out0", "leakyrelu:out0"]],
"param_map": {'leakyrelu':{'leaky_ratio':['FLOAT', 'CODE',"self.attr_pick('aten::leaky_relu', 'a_1')"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_leakrelu)


r_contiguous_2_noop = {
"ruler_name": "r_clone_2_noop",
"src_ops_alias": ["aten::contiguous"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::contiguous:in0"]],
"src_out_tensor": ["aten::contiguous:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["aten::contiguous:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_contiguous_2_noop)


r_upsample_bilinear2d = {
"ruler_name": "r_upsample_bilinear2d",
"src_ops_alias": ["aten::upsample_bilinear2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::upsample_bilinear2d:in0"]],
"src_out_tensor": ["aten::upsample_bilinear2d:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["aten::upsample_bilinear2d:out0", "image_resize:out0"]],
"param_map": {'image_resize':{
    'type': ['STRING', 'VALUE', "bilinear"],
    'new_size': ['INTS', 'CODE', "self.attr_pick('aten::upsample_bilinear2d', 'a_1')"],
    'align_corners': ['BOOL', 'CODE', "self.attr_pick('aten::upsample_bilinear2d', 'a_2')"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_upsample_bilinear2d)

r_tanh = {
"ruler_name": "r_tanh",
"src_ops_alias": ["aten::tanh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::tanh:in0"]],
"src_out_tensor": ["aten::tanh:out0"],
"acu_lys_alias": ["tanh"],
"src_acu_in_tensor_map": [["I:out0", "tanh:in0"]],
"src_acu_out_tensor_map": [["aten::tanh:out0", "tanh:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_tanh)

r_reshape = {
"ruler_name": "r_reshape",
"src_ops_alias": ["aten::reshape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::reshape:in0"]],
"src_out_tensor": ["aten::reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["aten::reshape:out0", "reshape:out0"]],
"param_map": {'reshape':{'shape': ['INTS', 'CODE', "self.reshape_shape(self.attr_pick('aten::reshape', 'a_1'))"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_reshape)

r_softmax = {
"ruler_name": "r_softmax",
"src_ops_alias": ["aten::softmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::softmax:in0"]],
"src_out_tensor": ["aten::softmax:out0"],
"acu_lys_alias": ["softmax"],
"src_acu_in_tensor_map": [["I:out0", "softmax:in0"]],
"src_acu_out_tensor_map": [["aten::softmax:out0", "softmax:out0"]],
"param_map": {'softmax':{'sf_axis': ['ORIGIN', 'CODE', "self.attr_pick('aten::softmax', 'a_1')"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_softmax)

r_permute = {
"ruler_name": "r_permute",
"src_ops_alias": ["aten::permute"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::permute:in0"]],
"src_out_tensor": ["aten::permute:out0"],
"acu_lys_alias": ["permute"],
"src_acu_in_tensor_map": [["I:out0", "permute:in0"]],
"src_acu_out_tensor_map": [["aten::permute:out0", "permute:out0"]],
"param_map": {
    'permute':{'perm': ['STRING', 'CODE', "' '.join([str(p) for p in self.attr_pick('aten::permute', 'a_1')])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_permute)

r_clone_2_noop = {
"ruler_name": "r_clone_2_noop",
"src_ops_alias": ["aten::clone"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::clone:in0"]],
"src_out_tensor": ["aten::clone:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["aten::clone:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_clone_2_noop)

r_log_softmax = {
"ruler_name": "r_log_softmax",
"src_ops_alias": ["aten::log_softmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::log_softmax:in0"]],
"src_out_tensor": ["aten::log_softmax:out0"],
"acu_lys_alias": ["softmax", "log"],
"src_acu_in_tensor_map": [["I:out0", "softmax:in0"]],
"src_acu_out_tensor_map": [["aten::log_softmax:out0", "log:out0"]],
"param_map": {'softmax':{'sf_axis':['INT', 'CODE',"self.attr_pick('aten::log_softmax', 'a_1')"],},
              'log':{}},
"blob_map": {},
"acu_inter_flow": [["softmax:out0", "log:in0"]],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_log_softmax)

r_view_as_reshape = {
"ruler_name": "r_view_as_reshape",
"src_ops_alias": ["aten::view"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::view:in0"]],
"src_out_tensor": ["aten::view:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["aten::view:out0", "reshape:out0"]],
"param_map": {'reshape':{'shape':['INTS', 'CODE',"self.reshape_shape(self.attr_pick('aten::view', 'a_1'))"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_view_as_reshape)

r_threshold_relu = {
"ruler_name": "r_threshold_relu",
"src_ops_alias": [r"aten::threshold"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::threshold:in0"]],
"src_out_tensor": [r"aten::threshold:out0"],
"acu_lys_alias": ["relu"],
"src_acu_in_tensor_map": [["I:out0", "relu:in0"]],
"src_acu_out_tensor_map": [[r"aten::threshold:out0", "relu:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick('aten::threshold', 'a_1') == 0 and self.attr_pick('aten::threshold', 'a_2') == 0",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_threshold_relu)

r_elu = {
"ruler_name": "r_elu",
"src_ops_alias": [r"aten::elu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::elu:in0"]],
"src_out_tensor": [r"aten::elu:out0"],
"acu_lys_alias": ["elu"],
"src_acu_in_tensor_map": [["I:out0", "elu:in0"]],
"src_acu_out_tensor_map": [[r"aten::elu:out0", "elu:out0"]],
"param_map": {'elu':{'alpha':['FLOAT', 'CODE',"self.attr_pick('aten::elu', 'a_1')"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick('aten::elu', 'a_1') == 0 and self.attr_pick('aten::elu', 'a_2') == 0",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_elu)

r_hardtanh_ = {
"ruler_name": "r_hardtanh_",
"src_ops_alias": ["aten::hardtanh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::hardtanh:in0"]],
"src_out_tensor": ["aten::hardtanh:out0"],
"acu_lys_alias": ["tanh", "relun"],
"src_acu_in_tensor_map": [["I:out0", "tanh:in0"]],
"src_acu_out_tensor_map": [["aten::hardtanh:out0", "relun:out0"]],
"acu_inter_flow": [["tanh:out0", "relun:in0"]],
"param_map": {'relun':{'relu_clamp_bottom':['FLOAT', 'CODE',"self.attr_pick('aten::hardtanh', 'a_1')"],
                       'relu_clamp_top':['FLOAT', 'CODE',"self.attr_pick('aten::hardtanh', 'a_2')"],},
              'tanh':{}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_hardtanh_)

r_add = {
"ruler_name": "r_add",
"src_ops_alias": [r"aten::add"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::add:in0"], ["I_1:out0", r"aten::add:in1"]],
"src_out_tensor": [r"aten::add:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map": [["I:out0", "add:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [[r"aten::add:out0", "add:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
# a_1 stand for alpha of add operation.
"pre_condition": "self.attr_pick('aten::add', 'a_2') == 1",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_add)


r_pixel_shuffle = {
"ruler_name": "r_pixel_shuffle",
"src_ops_alias": [r"aten::pixel_shuffle"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::pixel_shuffle:in0"]],
"src_out_tensor": [r"aten::pixel_shuffle:out0"],
"acu_lys_alias": ["permute_0", "reshape", "permute", "reshape_1", "permute_1"],
"src_acu_in_tensor_map": [["I:out0", "permute_0:in0"]],
"src_acu_out_tensor_map": [[r"aten::pixel_shuffle:out0", "permute_1:out0"]],
"param_map": {
    "permute_0":{"perm":["STRING", "VALUE", "0 3 1 2"]},
    "reshape":{"shape":["INTS", "CODE", "self.pixel_shuffle_reshape_step1('aten::pixel_shuffle')"]},
    "permute":{"perm": ["STRING", "VALUE", "0 1 4 2 5 3"]},
    "reshape_1":{'shape':["INTS", "CODE", "self.pixel_shuffle_reshape_step2('aten::pixel_shuffle')"]},
    "permute_1":{'perm':["STRING", "VALUE", "0 2 3 1"]},
},
"blob_map": {},
"acu_inter_flow": [["permute_0:out0","reshape:in0"],
                   ["reshape:out0", "permute:in0"],
                   ["permute:out0", "reshape_1:in0"],
                   ["reshape_1:out0", "permute_1:in0"]],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_pixel_shuffle)

def r_cat_templete_fun(cat_count):
    r_concat_dict = {
"ruler_name": "r_cat_{}".format(cat_count),
"src_ops_alias": ["aten::cat", "prim::ListConstruct"],
"src_inter_flow": [["prim::ListConstruct:out0","aten::cat:in0"]],
"src_in_anchor": [["I_{}:out0".format(port), "prim::ListConstruct:in{}".format(port)] for port in range(cat_count)],
"src_out_tensor": ["aten::cat:out0"],
"acu_lys_alias": ["concat"],
"src_acu_in_tensor_map": [["I_{}:out0".format(order), "concat:in{}".format(order)] for order in range(cat_count)],
"src_acu_out_tensor_map": [["aten::cat:out0", "concat:out0"]],
"param_map": {"concat": {"dim": ["INT", "CODE", "self.attr_pick('aten::cat', 'a_1')"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
    return r_concat_dict
# ruler_list.extend([r_cat_templete_fun(count) for count in range(2, 30)])

r_cat_2_noop = {
"ruler_name": "r_cat_2_noop",
"src_ops_alias": ["aten::cat", "prim::ListConstruct"],
"src_inter_flow": [["prim::ListConstruct:out0", "aten::cat:in0"]],
"src_in_anchor": [["I:out0", "prim::ListConstruct:in0"]],
"src_out_tensor": ["aten::cat:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["aten::cat:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_cat_2_noop)

r_batchnorm = {
    "ruler_name": "r_batchnorm",
    "src_ops_alias": ["aten::batch_norm", "const_1", "const_2", "const_3", "const_4"],
    "src_inter_flow": [["const_1:out0","aten::batch_norm:in1"],
                       ["const_2:out0","aten::batch_norm:in2"],
                       ["const_3:out0","aten::batch_norm:in3"],
                       ["const_4:out0","aten::batch_norm:in4"]],
    "src_in_anchor": [["I:out0", "aten::batch_norm:in0"]],
    "src_out_tensor": ["aten::batch_norm:out0"],
    "acu_lys_alias": ["batchnormalize"],
    "src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
    "src_acu_out_tensor_map": [["aten::batch_norm:out0", "batchnormalize:out0"]],
    "param_map": {"batchnormalize":
                      {"eps": ["FLOAT", "CODE", "self.attr_pick('aten::batch_norm', 'a_7')"]}},
    "blob_map":
        {"batchnormalize":
             {"gamma": ["CODE", "self.tensor_to_numpy('aten::batch_norm:in1')"],
              "beta": ["CODE", "self.tensor_to_numpy('aten::batch_norm:in2')"],
              "variance": ["CODE", "self.tensor_to_numpy('aten::batch_norm:in4')"],
              "mean": ["CODE", "self.tensor_to_numpy('aten::batch_norm:in3')"]}},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None,
    "src_ops_main_version": [1, -1],
    "src_ops_minior_version": [0, -1]}
ruler_list.append(r_batchnorm)

r_conv = {
"ruler_name": "r_conv",
"src_ops_alias": ["aten::_convolution"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::_convolution:in0"]],
"src_out_tensor": ["aten::_convolution:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["aten::_convolution:out0", "convolution:out0"]],
"param_map":{'convolution':
                  {"padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "bias": ["BOOL", "CODE", "isinstance(self.tensor_to_numpy('aten::_convolution:in2'), np.ndarray)"],
                   "group_number": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_8')"],
                   "weights": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[0]"],
                   "ksize_h": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[2]"],
                   "ksize_w": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[3]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[0]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[1]"],
                   "pad": ["INTS", "CODE", "self.map_pad_value(self.attr_pick('aten::_convolution', 'a_4'))"],
                   }
             },
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy('aten::_convolution:in1', [2, 3, 1, 0])"],
                  "bias": ["CODE", "self.tensor_to_numpy('aten::_convolution:in2')"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.tensor_shape('aten::_convolution:in1')) == 4 and "
                 "isinstance(self.tensor_to_numpy('aten::_convolution:in1'), np.ndarray) == True and "
                 "self.attr_pick('aten::_convolution', 'a_6') == False",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_conv)

r_trans_conv = {
"ruler_name": "r_trans_conv",
"src_ops_alias": ["aten::_convolution"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::_convolution:in0"]],
"src_out_tensor": ["aten::_convolution:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["aten::_convolution:out0", "deconvolution:out0"]],
"param_map":{'deconvolution':
                  {"padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "bias": ["BOOL", "CODE", "isinstance(self.tensor_to_numpy('aten::_convolution:in2'), np.ndarray)"],
                   "group_number": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_8')"],
                   "weights": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[0]"],
                   "ksize_h": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[2]"],
                   "ksize_w": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[3]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[0]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[1]"],
                   "pad": ["INTS", "CODE", "self.map_pad_value(self.attr_pick('aten::_convolution', 'a_4'))"],
                   # "output_shape": ['INTS', 'CODE',
                   #                           "self.tensor_shape('aten::_convolution:in0')"]
                   },

             },
"blob_map": {"deconvolution":
                 {"weight": ["CODE", "self.tensor_to_numpy('aten::_convolution:in1', [2, 3, 1, 0])"],
                  "bias": ["CODE", "self.tensor_to_numpy('aten::_convolution:in2')"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.tensor_shape('aten::_convolution:in1')) == 4 and "
                 "isinstance(self.tensor_to_numpy('aten::_convolution:in1'), np.ndarray) == True and "
                 "self.attr_pick('aten::_convolution', 'a_6') == True",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_trans_conv)

r_conv2dOP = {
"ruler_name": "r_conv2dOP",
"src_ops_alias": ["aten::_convolution", "aten::view"],
"src_inter_flow": [["aten::view:out0","aten::_convolution:in1"]],
"src_in_anchor": [["I:out0", "aten::_convolution:in0"],["I_1:out0", "aten::view:in0"]],
"src_out_tensor": ["aten::_convolution:out0"],
"acu_lys_alias": ["conv2d_op", "reshape"],
"src_acu_in_tensor_map": [["I:out0", "conv2d_op:in0"], ["I_1:out0", "reshape:in0"]],
"acu_inter_flow": [["reshape:out0", "conv2d_op:in1"]],
"src_acu_out_tensor_map": [["aten::_convolution:out0", "conv2d_op:out0"]],
"param_map":{'conv2d_op':
                  {"padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "group_number": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_8')"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[0]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[1]"],
                   "pad": ["INTS", "CODE", "self.map_pad_value(self.attr_pick('aten::_convolution', 'a_4'))"],
                   },
             'reshape':{'shape':['INTS', 'CODE',"self.array_layout(self.attr_pick('aten::view', 'a_1'), [2,3,0,1])"],}
             },
"blob_map": {"conv2d_op":{}},
"priority_tip": 0,
"pre_condition": "len(self.tensor_shape('aten::_convolution:in1')) == 4 and "
                 "isinstance(self.tensor_to_numpy('aten::_convolution:in1'), np.ndarray) == False and "
                 "isinstance(self.tensor_to_numpy('aten::_convolution:in2'), np.ndarray) == False",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_conv2dOP)

r_conv3d = {
"ruler_name": "r_conv3d",
"src_ops_alias": ["aten::_convolution"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::_convolution:in0"]],
"src_out_tensor": ["aten::_convolution:out0"],
"acu_lys_alias": ["conv3d"],
"src_acu_in_tensor_map": [["I:out0", "conv3d:in0"]],
"src_acu_out_tensor_map": [["aten::_convolution:out0", "conv3d:out0"]],
"param_map":{'conv3d':
                  {"padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "bias": ["BOOL", "CODE", "isinstance(self.tensor_to_numpy('aten::_convolution:in2'), np.ndarray)"],
                   "group_number": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_8')"],
                   "weights": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[0]"],
                   "ksize_d": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[2]"],
                   "ksize_h": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[3]"],
                   "ksize_w": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[4]"],
                   "stride_d": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[0]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[1]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[2]"],
                   "pad": ["INTS", "CODE", "self.map_pad_value(self.attr_pick('aten::_convolution', 'a_4'))"],
                   }
             },
"blob_map": {"conv3d":
                 {"weight": ["CODE", "self.tensor_to_numpy('aten::_convolution:in1', [2, 3, 4, 1, 0])"],
                  "bias": ["CODE", "self.tensor_to_numpy('aten::_convolution:in2')"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.tensor_shape('aten::_convolution:in1')) == 5",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_conv3d)

r_conv_zero_bias = {
"ruler_name": "r_conv_zero_bias",
"src_ops_alias": ["aten::_convolution", "aten::zeros", "aten::add"],
"src_inter_flow": [["aten::zeros:out0", "aten::add:in0"], ["aten::_convolution:out0", "aten::add:in1"]],
"src_in_anchor": [["I:out0", "aten::_convolution:in0"]],
"src_out_tensor": ["aten::add:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["aten::add:out0", "convolution:out0"]],
"param_map":{'convolution':
                  {"padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "bias": ["BOOL", "CODE", "isinstance(self.tensor_to_numpy('aten::_convolution:in2'), np.ndarray)"],
                   "group_number": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_8')"],
                   "weights": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[0]"],
                   "ksize_h": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[2]"],
                   "ksize_w": ["INT", "CODE", "self.tensor_shape('aten::_convolution:in1')[3]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[0]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::_convolution', 'a_3')[1]"],
                   "pad": ["INTS", "CODE", "self.map_pad_value(self.attr_pick('aten::_convolution', 'a_4'))"],
                   }
             },
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy('aten::_convolution:in1', [2, 3, 1, 0])"],
                  "bias": ["CODE", "self.tensor_to_numpy('aten::_convolution:in2')"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_conv_zero_bias)

r_dcon = {
"ruler_name": "r_dcon",
"src_ops_alias": ["aten::conv_transpose2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::conv_transpose2d:in0"]],
"src_out_tensor": ["aten::conv_transpose2d:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["aten::conv_transpose2d:out0", "deconvolution:out0"]],
"param_map":{'deconvolution':
                  {"padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "bias": [
                       "BOOL", "CODE", "isinstance(self.tensor_to_numpy('aten::conv_transpose2d:in2'), np.ndarray)"],
                   "group_number": ["INT", "CODE", "self.attr_pick('aten::conv_transpose2d', 'a_8')"],
                   "weights": ["INT", "CODE", "self.tensor_shape('aten::conv_transpose2d:in1')[0]"],
                   "ksize_h": ["INT", "CODE", "self.tensor_shape('aten::conv_transpose2d:in1')[2]"],
                   "ksize_w": ["INT", "CODE", "self.tensor_shape('aten::conv_transpose2d:in1')[3]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::conv_transpose2d', 'a_3')[0]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::conv_transpose2d', 'a_3')[1]"],
                   "pad": ["INTS", "CODE", "self.map_pad_value(self.attr_pick('aten::conv_transpose2d', 'a_4'))"],
                   'output_shape': ['INTS', 'CODE',
                                             "self.deconv_output_shape(self.tensor_to_numpy(tensor['C:out0']))"]},
             },
"blob_map": {"deconvolution":
                 {"weight": ["CODE", "self.tensor_to_numpy('aten::conv_transpose2d:in1', [2, 3, 1, 0])"],
                  "bias": ["CODE", "self.tensor_to_numpy('aten::conv_transpose2d:in2')"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_dcon)

r_avg_pool = {
"ruler_name": "r_avg_pool",
"src_ops_alias": ["aten::avg_pool2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::avg_pool2d:in0"]],
"src_out_tensor": ["aten::avg_pool2d:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["aten::avg_pool2d:out0", "pooling:out0"]],
"param_map": {'pooling':
                  {"type": ["STRING", "VALUE", "AVG"],
                   "padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": [
                       "STRING",
                       "VALUE",
                       "padding_const"],
                   "ksize_h": ["INT", "CODE", "self.attr_pick('aten::avg_pool2d', 'a_1')[0]"],
                   "ksize_w": ["INT", "CODE", "self.attr_pick('aten::avg_pool2d', 'a_1')[1]"],
                   "stride_h": [
                       "INT",
                       "CODE",
                       "self.attr_pick('aten::avg_pool2d', 'a_2')[0]"
                       " if len(self.attr_pick('aten::avg_pool2d', 'a_2')) != 0 "
                       "else self.attr_pick('aten::avg_pool2d', 'a_1')[0]"],
                   "stride_w": [
                       "INT",
                       "CODE",
                       "self.attr_pick('aten::avg_pool2d', 'a_2')[1]"
                       " if len(self.attr_pick('aten::avg_pool2d', 'a_2')) != 0 "
                       "else self.attr_pick('aten::avg_pool2d', 'a_1')[1]"],
                   "pad":
                       ["INTS",
                        "CODE",
                        "self.map_pad_value(self.attr_pick('aten::avg_pool2d', 'a_3'))"],
                   "round_type":
                       ["STRING",
                        "CODE",
                        "'ceil' if self.attr_pick('aten::avg_pool2d', 'a_5') else 'floor'"
                        ],
                   }},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_avg_pool)

r_max_pool = {
"ruler_name": "r_max_pool",
"src_ops_alias": ["aten::max_pool2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::max_pool2d:in0"]],
"src_out_tensor": ["aten::max_pool2d:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["aten::max_pool2d:out0", "pooling:out0"]],
"param_map": {'pooling':
                  {"type": ["STRING", "VALUE", "MAX"],
                   "padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": [
                       "STRING",
                       "VALUE",
                       "padding_const"],
                   "ksize_h": ["INT", "CODE", "self.attr_pick('aten::max_pool2d', 'a_1')[0]"],
                   "ksize_w": ["INT", "CODE", "self.attr_pick('aten::max_pool2d', 'a_1')[1]"],
                   "stride_h": [
                       "INT",
                       "CODE",
                       "self.attr_pick('aten::max_pool2d', 'a_2')[0]"
                       " if len(self.attr_pick('aten::max_pool2d', 'a_2')) != 0 "
                       "else self.attr_pick('aten::max_pool2d', 'a_1')[0]"],
                   "stride_w": [
                       "INT",
                       "CODE",
                       "self.attr_pick('aten::max_pool2d', 'a_2')[1]"
                       " if len(self.attr_pick('aten::max_pool2d', 'a_2')) != 0 "
                       "else self.attr_pick('aten::max_pool2d', 'a_1')[1]"],
                   "pad":
                       ["INTS",
                        "CODE",
                        "self.map_pad_value(self.attr_pick('aten::max_pool2d', 'a_3'))"],
                   "round_type":
                       ["STRING",
                        "CODE",
                        "'ceil' if self.attr_pick('aten::max_pool2d', 'a_5') else 'floor'"
                        ],
                   }},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_max_pool)

r_max_pool_with_indices = {
"ruler_name": "r_max_pool_with_indices",
"src_ops_alias": ["aten::max_pool2d_with_indices"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::max_pool2d_with_indices:in0"]],
"src_out_tensor": ["aten::max_pool2d_with_indices:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["aten::max_pool2d_with_indices:out0", "pooling:out0"]],
"param_map": {'pooling':
                  {"type": ["STRING", "VALUE", "MAX"],
                   "padding": ["STRING", "VALUE", "VALID"],
                   "pad_method": [
                       "STRING",
                       "VALUE",
                       "padding_const"],
                   "ksize_h": ["INT", "CODE", "self.attr_pick('aten::max_pool2d_with_indices', 'a_1')[0]"],
                   "ksize_w": ["INT", "CODE", "self.attr_pick('aten::max_pool2d_with_indices', 'a_1')[1]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::max_pool2d_with_indices', 'a_2')[0]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::max_pool2d_with_indices', 'a_2')[1]"],
                   "pad":
                       ["INTS",
                        "CODE",
                        "self.map_pad_value(self.attr_pick('aten::max_pool2d_with_indices', 'a_3'))"],
                   "round_type":
                       ["STRING",
                        "CODE",
                        "'ceil' if self.attr_pick('aten::max_pool2d_with_indices', 'a_5') else 'floor'"
                        ],
                   }},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_max_pool_with_indices)

r_max_pool3d_with_indices = {
"ruler_name": "r_max_pool3d_with_indices",
"src_ops_alias": ["aten::max_pool3d_with_indices"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::max_pool3d_with_indices:in0"]],
"src_out_tensor": ["aten::max_pool3d_with_indices:out0"],
"acu_lys_alias": ["pool3d"],
"src_acu_in_tensor_map": [["I:out0", "pool3d:in0"]],
"src_acu_out_tensor_map": [["aten::max_pool3d_with_indices:out0", "pool3d:out0"]],
"param_map": {'pool3d':
                  {"type": ["STRING", "VALUE", "MAX"],
                   "pad_method": ["STRING", "VALUE", "padding_const"],
                   "ksize_d": ["INT", "CODE", "self.attr_pick('aten::max_pool3d_with_indices', 'a_1')[0]"],
                   "ksize_h": ["INT", "CODE", "self.attr_pick('aten::max_pool3d_with_indices', 'a_1')[1]"],
                   "ksize_w": ["INT", "CODE", "self.attr_pick('aten::max_pool3d_with_indices', 'a_1')[2]"],
                   "stride_d": ["INT", "CODE", "self.attr_pick('aten::max_pool3d_with_indices', 'a_2')[0]"],
                   "stride_h": ["INT", "CODE", "self.attr_pick('aten::max_pool3d_with_indices', 'a_2')[1]"],
                   "stride_w": ["INT", "CODE", "self.attr_pick('aten::max_pool3d_with_indices', 'a_2')[2]"],
                   "pad":
                       ["INTS",
                        "CODE",
                        "self.map_pad_value(self.attr_pick('aten::max_pool3d_with_indices', 'a_3'))"],
                   "round_type":
                       ["STRING",
                        "CODE",
                        "'ceil' if self.attr_pick('aten::max_pool3d_with_indices', 'a_5') else 'floor'"
                        ],
                   }},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_max_pool3d_with_indices)

r_adaptive_avg_pool2d_2_imgresize = {
"ruler_name": "r_adaptive_avg_pool2d_2_imgresize",
"src_ops_alias": ["aten::adaptive_avg_pool2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::adaptive_avg_pool2d:in0"]],
"src_out_tensor": ["aten::adaptive_avg_pool2d:out0"],
"acu_lys_alias": ["resizebilinear_image"],
"src_acu_in_tensor_map": [["I:out0", "resizebilinear_image:in0"]],
"src_acu_out_tensor_map": [["aten::adaptive_avg_pool2d:out0", "resizebilinear_image:out0"]],
"param_map": {'resizebilinear_image': \
    {'new_size':['INTS', 'CODE',"self.attr_pick('aten::adaptive_avg_pool2d', 'a_1')"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_adaptive_avg_pool2d_2_imgresize)

r_adaptive_avg_pool2d_2_noop = {
"ruler_name": "r_adaptive_avg_pool2d_2_noop",
"src_ops_alias": ["aten::adaptive_avg_pool2d"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::adaptive_avg_pool2d:in0"]],
"src_out_tensor": ["aten::adaptive_avg_pool2d:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["aten::adaptive_avg_pool2d:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": \
    "self.attr_pick('aten::adaptive_avg_pool2d', 'a_1') == self.tensor_shape('aten::adaptive_avg_pool2d:in0')",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_adaptive_avg_pool2d_2_noop)

r_flatten = {
"ruler_name": "r_flatten",
"src_ops_alias": ["aten::flatten"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::flatten:in0"]],
"src_out_tensor": ["aten::flatten:out0"],
"acu_lys_alias": ["flatten", "permute"],
"src_acu_in_tensor_map": [["I:out0", "permute:in0"]],
"src_acu_out_tensor_map": [["aten::flatten:out0", "flatten:out0"]],
"param_map": {'flatten':{'axis':['INT', 'CODE',"self.attr_pick('aten::flatten', 'a_1')"],},
              'permute':{'perm': ['STRING', 'VALUE', '0 1 2 3']}},
"blob_map": {},
"acu_inter_flow": [["permute:out0", "flatten:in0"]],
"priority_tip": 0,
"pre_condition": "len(self.tensor_shape('aten::flatten:in0')) == 4" \
    + " and self.attr_pick('aten::flatten', 'a_1') == 1 and self.attr_pick('aten::flatten', 'a_2') == -1",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_flatten)

r_dropout = {
"ruler_name": "r_dropout",
"src_ops_alias": ["aten::dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::dropout:in0"]],
"src_out_tensor": ["aten::dropout:out0"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["aten::dropout:out0", "dropout:out0"]],
"param_map": {'dropout':{'ratio':['FLOAT', 'CODE',"self.attr_pick('aten::dropout', 'a_1')"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick('aten::dropout', 'a_2') == True",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_dropout)

r_dropout_2_noop = {
"ruler_name": "r_dropout_2_noop",
"src_ops_alias": ["aten::dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::dropout:in0"]],
"src_out_tensor": ["aten::dropout:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["aten::dropout:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick('aten::dropout', 'a_2') == False",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_dropout_2_noop)

r_feature_dropout_2_noop = {
"ruler_name": "r_feature_dropout_2_noop",
"src_ops_alias": ["aten::feature_dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "aten::feature_dropout:in0"]],
"src_out_tensor": ["aten::feature_dropout:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["aten::feature_dropout:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick('aten::feature_dropout', 'a_2') == False",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_feature_dropout_2_noop)

r_relu = {
"ruler_name": "r_relu",
"src_ops_alias": [r"aten::relu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", r"aten::relu:in0"]],
"src_out_tensor": [r"aten::relu:out0"],
"acu_lys_alias": ["relu"],
"src_acu_in_tensor_map": [["I:out0", "relu:in0"]],
"src_acu_out_tensor_map": [[r"aten::relu:out0", "relu:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_relu)

r_addmm = {
"ruler_name": "r_addmm",
"src_ops_alias": ["aten::addmm", "aten::t", "const_1", "const_2"],
"src_inter_flow": \
    [["aten::t:out0", "aten::addmm:in2"], ["const_1:out0", "aten::t:in0"], ["const_2:out0", "aten::addmm:in0"]],
"src_in_anchor": [["I:out0", "aten::addmm:in1"]],
"src_out_tensor": ["aten::addmm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["aten::addmm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.tensor_shape('const_1:out0')[0]"],
                   "bias": ["BOOL", "CODE", "isinstance(self.tensor_to_numpy('const_2:out0'), np.ndarray)"]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.tensor_to_numpy('const_1:out0', [1, 0])"],
                  "bias": ["CODE", "self.tensor_to_numpy('const_2:out0')"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick('aten::addmm', 'a_3') == 1 "\
    "and self.attr_pick('aten::addmm', 'a_4') == 1",
"src_ops_main_version": [1, -1],
"src_ops_minior_version": [0, -1]}
ruler_list.append(r_addmm)

# r_const_2_variable = {
#     "ruler_name": "r_const_2_variable",
#     "src_ops_alias": ["prim::Constant"],
#     "src_inter_flow": [],
#     "src_in_anchor": [],
#     "src_out_tensor": ["prim::Constant:out0"],
#     "acu_lys_alias": ["variable"],
#     "src_acu_in_tensor_map": [],
#     "src_acu_out_tensor_map": [["prim::Constant:out0", "variable:out0"]],
#     "param_map": {"variable":
#                       {"shape": ["INTS", "CODE", "[1] "\
#                                                  "if isinstance(self.tensor_shape('prim::Constant:out0'), int) "\
#                                                  "else self.tensor_shape('prim::Constant:out0')"]}},
#     "blob_map": {"variable":
#                      {'data':
#                           ['CODE',
#                            "np.array([self.tensor_to_numpy('prim::Constant:out0')], dtype=np.float32) " \
#                            " if isinstance(self.tensor_shape('prim::Constant:out0'), int) " \
#                            "else self.tensor_to_numpy('prim::Constant:out0', cast_to='float32')"], }},
#     "acu_inter_flow": [],
#     "priority_tip": 0,
#     "pre_condition": None,
#     "src_ops_main_version": [1, -1],
#     "src_ops_minior_version": [0, -1]}
# ruler_list.append(r_const_2_variable)
#
#
#

def gen_pytorch_ruler(dst_path):
    # print(json.dumps(ruler_list))
    dst_path = os.path.join(dst_path, 'pytorch_ir_ruler_db.json')

    with open(dst_path, 'w+') as f:
        json.dump(ruler_list, f, indent=1)

    # To Verify ruler follow synatx
    with open(dst_path, 'r') as f:
        x_val = json.load(f)
def main():
    gen_pytorch_ruler(sys.argv[1])

if  __name__ == '__main__':
    main()

