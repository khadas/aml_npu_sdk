from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.core.tensor import Tensor
from acuitylib.xtf import xtf as tf
from acuitylib.utils import convert_axis_if_need,print_tensor
from acuitylib.acuitylog import AcuityLog as al
import numpy as np

from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.xtf import xtf as tf

class InterpLayer(CustomLayer):
    op = 'interp'

    # label, description
    def_input = [IoMap('in0', 'in', 'input port'), IoMap('in1', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def setup(self, inputs, outputs):

        out_shape = Shape()
        if self.net.get_platform_mode() == "nhwc":
            p = self.params
            in_shape1 = inputs[1].shape.dims
            p.height = in_shape1[1]
            p.width = in_shape1[2]
            out_shape.dims = [in_shape1[0], in_shape1[1], in_shape1[2], in_shape1[3]]
            outputs[0].shape = out_shape

        elif self.net.get_platform_mode() == "nchw":
            p = self.params
            in_shape1 = inputs[1].shape.dims
            p.height = in_shape1[2]
            p.width = in_shape1[3]
            out_shape.dims = [in_shape1[0], in_shape1[1], in_shape1[2], in_shape1[3]]
            outputs[0].shape = out_shape


    def _np_interp(self,input_tensor0):
        input = input_tensor0
        in_height = input.shape[1]
        in_width = input.shape[2]
        out_shape = self.get_output().shape.dims
        batch = out_shape[0]
        output_height = out_shape[1]
        output_width = out_shape[2]
        channel = out_shape[3]

        rheight = float((in_height - 1) / (output_height - 1))
        rwidth  = float((in_width - 1) / (output_width - 1))
        out = np.zeros([batch,output_height,output_width,channel],np.float32)
        #print('rheight, rwidth, c \n',rheight, rwidth)
        #print('in_width, in_height, c \n',in_width, in_height)
        #print('output_width, output_height, c \n',output_width, output_height)
        for h2 in range(0,output_height):
            h1r = rheight * h2
            h1 = int(h1r)
            h1p = 1 if (h1 < (in_height - 1)) else 0
            h1lambda = float(h1r - h1)
            h0lambda = float(1.0 - h1lambda)
            for w2 in range(0,output_width):
                w1r = rwidth * w2
                w1 = int(w1r)
                w1p = 1 if (w1 < (in_width - 1)) else 0
                w1lambda = float(w1r - w1)
                w0lambda = float(1.0 - w1lambda)
                #print('h1, w1, c \n',h1, w1, channel)
                #print('h2, w2, c \n',h2, w2, channel)
                #print('w1p, h1p, c \n',w1p, h1p, channel)
                for c in range(0,channel):
                    out[0,h2,w2,c] = h0lambda * (w0lambda * input[0,h1,w1,c] + w1lambda * input[0,h1,w1+w1p,c]) + \
                    h1lambda * (w0lambda * input[0,h1+h1p,w1,c] + w1lambda * input[0,h1+h1p,w1+w1p,c])
        return out

    def compute_out_tensor(self, tensor, input_tensor):
        shape0 = input_tensor[0].get_shape().as_list()
        shape1 = input_tensor[1].get_shape().as_list()
        print("run-compute",self.net.get_platform_mode())
        if ((shape0[2] == shape1[2]) and (shape0[1] == shape1[1])):
           out = input_tensor[0]
           return [out]

        out = tf.numpy_function(self._np_interp, [input_tensor[0]], [tf.float32])

        batch = shape0[0]
        out_shape = self.get_output(0).shape.dims
        out_shape[0] = batch

        return out # out is already a list in function self._np_interp, no need to add []