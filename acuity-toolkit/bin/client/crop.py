from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.xtf import xtf as tf

class Crop(CustomLayer):

    op = 'crop'

    # label, description
    def_output = [IoMap('out0', 'out', 'output port')]

    def setup(self, inputs, outputs):
        p = self.params
        size = inputs[0].shape.dims
        crop = inputs[1].shape.dims
        shape = size[:p.axis] + crop[p.axis:]
        outputs[0].shape = Shape(shape)

    def compute_out_tensor(self, tensor, input_tensor):
        p = self.params
        img = input_tensor[0]
        org_size = [s for s in img.get_shape().as_list()]
        crop = [s for s in input_tensor[1].get_shape().as_list()]
        crop = org_size[:p.axis] + crop[p.axis:]
        tf_size = crop
        if self.net.get_platform_mode() == 'nchw':
            size = [tf_size[0], tf_size[3], tf_size[1], tf_size[2]]
            org_size = [org_size[0], org_size[3], org_size[1], org_size[2]]
        else:
            size = tf_size
        if len(p.offset) == 1:
            begin = [0]*p.axis + (p.offset * (len(size) - p.axis))
        elif len(p.offset) == 0:
            begin = [0] * p.axis + ([0] * (len(size) - p.axis))
        else:
            begin = [0]*p.axis + p.offset
        size = org_size[:p.axis] + size[p.axis:]
        if self.net.get_platform_mode() == 'nchw':
            tf_size = [size[0], size[2], size[3], size[1]]
            tf_begin = [begin[0], begin[2], begin[3], begin[1]]
        else:
            tf_begin = begin
        out = tf.slice(img, tf_begin, tf_size)
        return [out]

    def load_params_from_caffe(self, cl):
        p = dict()
        p['axis'] = getattr(cl.crop_param, 'axis', 2)#caffe default axis is 2
        p['offset'] =  list(getattr(cl.crop_param, 'offset', 0))
        self.set_params(p)
