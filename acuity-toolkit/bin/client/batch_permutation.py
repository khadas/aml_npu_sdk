from acuitylib.layer.customlayer import CustomLayer
from acuitylib.core.shape import Shape
import numpy as np
from acuitylib.xtf import xtf as tf

class BatchPermutation(CustomLayer):

    op = 'batch_permutation'

    def _batch_permutation(self, tensor, inds):
        inds = np.reshape(inds, [-1])
        out = tensor[inds,:]
        return out

    def setup(self, inputs, outputs):
        shape = inputs[0].shape.dims
        shape[0] = inputs[1].shape.dims[0]
        outputs[0].shape = Shape(shape)

    def compute_out_tensor(self, tensor, input_tensor):
        outs = tf.numpy_function(self._batch_permutation, input_tensor, [tf.float32])
        outs[0].set_shape(self.get_output(0).shape.dims)
        return outs


