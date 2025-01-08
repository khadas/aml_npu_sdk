from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.xtf import xtf as tf
import numpy as np

class ResizeSqrt(CustomLayer):

    op = 'resizesqrt'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def setup(self, inputs, outputs):
        in_shape = inputs[0].shape
        s = np.array(np.round([in_shape.h, in_shape.w]/np.sqrt(2))).astype(np.int)
        outputs[0].shape = Shape([s[0], s[1], 1, 1])

    def compute_out_tensor(self, tensor, input_tensor):
        data = input_tensor[0]

        out_shape = self.get_out_shape().dims
        out_height = out_shape[0]
        out_width = out_shape[1]

        method=tf.image.ResizeMethod.BILINEAR
        out = tf.image.resize_images(data, [out_height, out_width], method)
        return [out]
