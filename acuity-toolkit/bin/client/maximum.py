from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.broadcast_layer import BroadcastLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.xtf import xtf as tf

class Maximum(CustomLayer):

    op = 'maximum'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port'), IoMap('in1', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]


    def setup(self, inputs, outputs):
        in_shape0 = inputs[0].shape.dims
        in_shape1 = inputs[1].shape.dims
        out_shape = BroadcastLayer.calc_broadcast_shape(in_shape0, in_shape1)
        shape = Shape(out_shape)
        outputs[0].shape = shape

    def compute_out_tensor(self, tensor, input_tensor):
        out = tf.maximum(input_tensor[0], input_tensor[1])
        return [out]