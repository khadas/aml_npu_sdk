from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.xtf import xtf as tf
import numpy as np

class Conv2DBackpropInput(CustomLayer):

    op = 'conv2dbackpropinput'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def get_variable_shape(self, coef):
        p = self.params
        shape = None
        if 'input_sizes' == coef:
            shape = self.get_const_tensor('input_sizes').const_data.shape
        elif 'filters' == coef:
            shape = self.get_const_tensor('filters').const_data.shape
        return shape

    def load_params_from_tf(self, ruler, layer_alias, op_alias_map, tensor_data_map, anet=None):
        # p = dict()
        # self.put_const_tensor('input_sizes', datas[0].astype(np.float32))
        # self.put_const_tensor('filters', datas[1].astype(np.float32))
        # p['strides'] = ','.join([str(i) for i in tl.attr['strides'].list.i])
        # p['padding'] = tl.attr['padding'].s.decode('utf-8')
        # self.set_params(p)
        pass


    def setup(self, inputs, outputs):
        outputs[0].shape = Shape(self.get_const_tensor('input_sizes').const_data.tolist())

    def compute_out_tensor(self, tensor, input_tensor):
        out = tf.compat.v1.nn.conv2d_backprop_input(self.get_const_tensor('input_sizes').const_data.astype(np.int32),
                                          self.get_const_tensor('filters').const_data,
                                          input_tensor[0],
                                          [int(s) for s in self.get_params()['strides'].split(',') ],
                                          self.get_params()['padding'])
        return [out]