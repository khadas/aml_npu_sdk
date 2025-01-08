import copy
from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.core.dtype import DType
from acuitylib.xtf import xtf as tf
from acuitylib.acuitylog import AcuityLog as al
from acuitylib.layer.layer_params import DefParam
import numpy as np

class ArgMax(CustomLayer):

    op = 'argmax'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def_param = [
        DefParam('output_type', DType.int64, False),
    ]

    def setup(self, inputs, outputs):
        in_shape = inputs[0].shape.dims
        out_shape = copy.deepcopy(in_shape)
        axis = self.params.axis
        if axis < 0:
            axis = len(in_shape) + axis
            setattr(self.params, 'axis', axis)
        out_shape.remove(in_shape[self.params.axis])
        outputs[0].shape = Shape(out_shape)

    def load_params_from_caffe(self, cl):
        p = dict()
        p['axis'] = cl.argmax_param.axis
        self.set_params(p)

    def load_params_from_tf(self, ruler, layer_alias, op_alias_map, tensor_data_map, anet=None):
        p = dict()
        p['axis'] = int(tensor_data_map['C:out0'])
        self.set_params(p)

    def load_params_from_onnx(self, node):
        p = dict()
        p['axis'] = 0
        if 'axis' in node.params:
            p['axis'] = node.params['axis']
        if 'keepdims' in node.params and node.params['keepdims'] == 1:
            al.w("Acuity don't support the keepdims = ture")
        if 'select_last_index' in node.params and node.params['select_last_index'] == 1:
            al.w("Acuity don't support the select_last_index = ture")
        self.set_params(p)

    def compute_out_tensor(self, tensor, input_tensor):
        p = self.params
        output_type = DType.map_to_backend_dtype(p.output_type, 'tensorflow')
        out = tf.argmax(input_tensor[0], self.params.axis, output_type=output_type)
        return [out]

    def compute_out_type(self):
        p = self.params
        self.get_output(0).type = p.output_type
