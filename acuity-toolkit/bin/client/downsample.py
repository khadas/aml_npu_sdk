from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.xtf import xtf as tf

class Downsample(CustomLayer):

    op = 'downsample'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def _down_sample(self, data):
        p = self.params
        out = data[::p.strides[0],::p.strides[1],::p.strides[2],::p.strides[3]]
        return out

    def setup(self, inputs, outputs):
        p = self.params
        shape = inputs[0].shape.dims
        shape[0] = int(shape[0]/p.strides[0])
        shape[1] = int(shape[1]/p.strides[1])
        shape[2] = int(shape[2]/p.strides[2])
        shape[3] = int(shape[3]/p.strides[3])
        outputs[0].shape = Shape(shape)

    def compute_out_tensor(self, tensor, input_tensor):
        out = tf.numpy_function(self._down_sample, [input_tensor[0]], [tf.float32])
        shape = self.get_output().shape.dims
        shape[0] = input_tensor[0].get_shape().as_list()[0]
        out[0].set_shape(shape)
        return out


