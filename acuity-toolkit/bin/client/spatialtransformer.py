from acuitylib.layer.customlayer import CustomLayer
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.core.shape import Shape
from acuitylib.core.tensor import Tensor
from acuitylib.utils import convert_axis_if_need,print_tensor
from acuitylib.xtf import xtf as tf
import numpy as np
from acuitylib.acuitylog import AcuityLog as al

class SpatialTransformer(CustomLayer):

    op = 'spatialtransformer'

    # label, description
    def_input  = [IoMap('in0', 'in', 'input port'),
                  IoMap('in1', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def _interpolate1(self, input_np, x_np, y_np):
        input = input_np
        x = x_np
        y = y_np
        out_shape = self.get_output().shape.dims

        batch = input.shape[0]
        height = input.shape[1]
        width = input.shape[2]
        channels = input.shape[3]
        out_height = out_shape[1]
        out_width = out_shape[2]

        height_f = float(height)
        width_f = float(width)

        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        x0 = np.floor(x)
        x1 = np.ceil(x)

        y0 = np.floor(y)
        y1 = np.ceil(y)

        x0 = x0.astype(np.int32)
        x1 = x1.astype(np.int32)
        y0 = y0.astype(np.int32)
        y1 = y1.astype(np.int32)

        out = np.zeros([batch,out_height,out_width,channels],np.float32)
        print(out.shape)
        for b in range(0,batch):
            for C in range(0,channels):
                for p in range(0, out_height * out_width):
                    m0 = x0[p]
                    m1 = x1[p]
                    n0 = y0[p]
                    n1 = y1[p]
                    value = 0.0

                    for m in range(m0,m1+1):
                        for n in range(n0,n1+1):
                            if m >= 0 and m < width and n >= 0 and n < height:
                                xvalue = x[p]
                                yvalue = y[p]
                                t = (1 - abs(xvalue - m)) * (1 - abs(yvalue - n))
                                t = t * input[b,m,n,C]
                                value = value + t
                    # out[b,height_idx,width_idx,C] =
                    # out[b,height_idx,height_idx,C] + (1 - abs(xvalue - m)) * (1 - abs(yvalue - n)) * C]
                    # calculate each point in loop
                    value = float(value)
                    width_idx = p % out_width
                    height_idx = int(p / out_width)
                    out[b, height_idx, width_idx, C] = value

        return out
    def _update_theta(self, data):
        theta_input = data[0]
        theta_input = np.reshape(theta_input, [theta_input.shape[0], -1])
        numParam = 0
        batch = theta_input.shape[1]
        width = theta_input.shape[0]

        if self.params.has_theta_1_1 == True:
            numParam = numParam + 1
        if self.params.has_theta_1_2 == True:
            numParam = numParam + 1
        if self.params.has_theta_1_3 == True:
            numParam = numParam + 1
        if self.params.has_theta_2_1 == True:
            numParam = numParam + 1
        if self.params.has_theta_2_2 == True:
            numParam = numParam + 1
        if self.params.has_theta_2_3 == True:
            numParam = numParam + 1
        if numParam + width != 6:
            al.e('The dimension of theta is not six!')

        if numParam == 0:
            out = data[0]
        else:
            out = np.zeros([6,batch],dtype=np.float32)
            theta_input_np = theta_input

            for b in range(0,batch):
                idx = 0
                if self.params.has_theta_1_1 == True:
                    out[0,b] = self.params.theta_1_1;
                else:
                    out[0,b] = theta_input_np[idx,b]
                    idx = idx + 1
                if self.params.has_theta_1_2 == True:
                    out[1,b] = self.params.theta_1_2;
                else:
                    out[1,b] = theta_input_np[idx,b]
                    idx = idx + 1
                if self.params.has_theta_1_3 == True:
                    out[2,b] = self.params.theta_1_3;
                else:
                    out[2,b] = theta_input_np[idx,b]
                    idx = idx + 1
                if self.params.has_theta_2_1 == True:
                    out[3,b] = self.params.theta_2_1;
                else:
                    out[3,b] = theta_input_np[idx,b]
                    idx = idx + 1
                if self.params.has_theta_2_2 == True:
                    out[4,b] = self.params.theta_2_2;
                else:
                    out[4,b] = theta_input_np[idx,b]
                    idx = idx + 1
                if self.params.has_theta_2_3 == True:
                    out[5,b] = self.params.theta_2_3;
                else:
                    out[5,b] = theta_input_np[idx,b]
        return [out]

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(self, im, x, y, out_size):
        # constants
        num_batch = im.get_shape().as_list()[0]
        height = im.get_shape().as_list()[1]
        width = im.get_shape().as_list()[2]
        channels = im.get_shape().as_list()[3]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[1]
        out_width = out_size[2]
        zero = tf.zeros_like(x, dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0
        # do sampling
        x1 = tf.cast(tf.ceil(x), 'int32')
        x0 = tf.cast(tf.floor(x), 'int32')
        #x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        #y1 = y0 + 1
        y1 = tf.cast(tf.ceil(y), 'int32')

        dim2 = width
        dim1 = width * height
        base = self._repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        zero = tf.zeros_like(Ia)
        Ia = tf.where(x0 < 0, x = zero, y = Ia)
        Ia = tf.where(x0 > max_x, x = zero, y = Ia)
        Ib = tf.where(x1 < 0, x = zero, y = Ib)
        Ib = tf.where(x1 > max_x, x = zero, y = Ib)
        Ic = tf.where(y0 < 0, x = zero, y = Ic)
        Ic = tf.where(y0 > max_y, x = zero, y = Ic)
        Id = tf.where(y1 < 0, x = zero, y = Id)
        Id = tf.where(y1 > max_x, x = zero, y = Id)
        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

    def _meshgrid(self, height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid

    def _transform(self, theta, input_dim, out_size):
        num_batch = input_dim.shape[0]
        height = input_dim.shape[2]
        width = input_dim.shape[3]
        num_channels = input_dim.shape[3]
        theta = theta.astype(np.float32)
        theta = np.reshape(theta,[-1,2,3])
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[2]
        out_width = out_size[3]

        grid_np = np.zeros([out_height * out_width, 3], np.float32)
        for m in range(0,out_height*out_width):
            grid_np[m, 0] = (m // out_width) * 1.0 / out_height * 2 - 1;
            grid_np[m, 1] = (m % out_width) * 1.0 / out_width * 2 - 1;
            grid_np[m, 2] = 1
        grid = grid_np.repeat(num_batch,axis=0)
        grid_1d = np.reshape(grid, [-1])
        grid = grid.reshape([num_batch, out_height*out_width, 3])
        grid = grid.transpose(0,2,1)
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = np.matmul(theta, grid)
        t_g_temp = T_g.transpose(0,2,1)
        t_g_1 = np.reshape(t_g_temp,[-1])
        x_s = T_g[:, 0:1, :]
        y_s = T_g[:, 1:, :]
        x_s_flat = np.reshape(x_s, [-1])
        y_s_flat = np.reshape(y_s, [-1])
        x = (x_s_flat+1.0) * height_f * 0.5
        y = (y_s_flat+1.0) * width_f * 0.5
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1
        x0 = np.clip(x0, 0, height - 1)
        x1 = np.clip(x1, 0, height - 1)
        y0 = np.clip(y0, 0, width - 1)
        y1 = np.clip(y1, 0, width - 1)
        Ia = input_dim[0][0][x0,y0]
        Ib = input_dim[0][0][x0,y1]
        Ic = input_dim[0][0][x1,y0]
        Id = input_dim[0][0][x1,y1]
        wa = (x1-x)*(y1-y)
        wb = (x1-x)*(y-y0)
        wc = (x-x0)*(y1-y)
        wd = (x-x0)*(y-y0)
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id
        out = np.reshape(out,out_size)
        out = out.astype(np.float32)
        out_temp = out.transpose(2,1,3,0)
        out_1 = np.reshape(out_temp, [-1])
        return out

    def setup(self, inputs, outputs):
        # in_shape = inputs[0].shape.dims
        # FIXME: remove convert_axis_if_need?
        # axis = convert_axis_if_need(self.net.get_platform_mode(),
        #         self.params.axis, inputs[0].shape.length())
        # outputs[0].shape = Shape(in_shape[:axis])
        out_shape = Shape()
        if self.net.get_platform_mode() == "nhwc":
            p = self.params
            shape = inputs[0].shape.dims
            if p.has_output_W == True:
                shape[2] = p.output_W
            if p.has_output_H == True:
                shape[1] = p.output_H
            p.output_H = shape[1]
            p.output_W = shape[2]
            out_shape.dims = [shape[0],shape[1],shape[2],shape[3]]
            outputs[0].shape = out_shape
        elif self.net.get_platform_mode() == "nchw":
            p = self.params
            shape = inputs[0].shape.dims
            if p.has_output_W == True:
                shape[3] = p.output_W
            if p.has_output_H == True:
                shape[2] = p.output_H
            p.output_H = shape[2]
            p.output_W = shape[3]
            out_shape.dims = [shape[0],shape[1],shape[2],shape[3]]
            outputs[0].shape = out_shape

        # shape_org = inputs[0].shape.dims
        # shape = [shape_org[0],shape_org[2],shape_org[3],shape_org[1]]



    def load_params_from_caffe(self, cl):
        p = dict()
        param = cl.st_param
        p['output_H'] = param.output_H
        p['output_W'] = param.output_W
        p['theta_1_2'] = param.theta_1_2
        p['theta_1_1'] = param.theta_1_1
        p['theta_1_3'] = param.theta_1_3
        p['theta_2_1'] = param.theta_2_1
        p['theta_2_2'] = param.theta_2_2
        p['theta_2_3'] = param.theta_2_3
        p['has_theta_1_1'] = param.HasField('theta_1_1')
        p['has_theta_1_2'] = param.HasField('theta_1_2')
        p['has_theta_1_3'] = param.HasField('theta_1_3')
        p['has_theta_2_1'] = param.HasField('theta_2_1')
        p['has_theta_2_2'] = param.HasField('theta_2_2')
        p['has_theta_2_3'] = param.HasField('theta_2_3')
        p['has_output_W'] = param.HasField('output_W')
        p['has_output_H'] = param.HasField('output_H')

        self.set_params(p)

    def compute_out_tensor(self, tensor, input_tensor):
        theta_list = tf.numpy_function(self._update_theta, [input_tensor[1]], [tf.float32])
        batch = input_tensor[1].get_shape().as_list()[0]
        width = input_tensor[1].get_shape().as_list()[1]
        theta = tf.reshape(theta_list[0], [batch, 2, 3])
        out_shape = self.get_output().shape.dims
        output = tf.numpy_function(self._transform, [theta, input_tensor[0], out_shape],[tf.float32])[0]
        output.set_shape(out_shape)
        return [output]
