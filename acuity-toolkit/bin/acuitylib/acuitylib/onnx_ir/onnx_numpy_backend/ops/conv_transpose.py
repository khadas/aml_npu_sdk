from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from acuitylib.xtf import xtf as tf
from acuitylib.acuitylog import AcuityLog as al

def get_output_shape(auto_pad,
              kernel_spatial_shape,
              input_spatial_shape,
              dilations,
              strides_spatial,
              pads=None,
              ):
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(
                        input_spatial_shape[i])
                    / float(
                        strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int((input_spatial_shape[i] - kernel_spatial_shape[i] - (kernel_spatial_shape[i] - 1) * (
                        dilations[i] - 1)) / float(strides_spatial[i]) + 1)
    return out_shape

def ConvTranspose(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    spatiall_rank = x.ndim - 2
    input_shape = np.shape(x)
    spatial_shape = input_shape[2:]

    w = inputs[1]

    # For onnx ConvTranspose:
    #   Input:  [batch, in_channel, in_h, in_w]
    #   Filter: [in_channel, out_channel/group, fsize_h, fsize_w]
    #   Bias:   [out_channel]

    auto_pad = attr.get('auto_pad', 'NOTSET')
    kernel_shape = attr.get('kernel_shape', w.shape[2:])
    dilations = attr.get('dilations', [1] * spatiall_rank)
    groups = attr.get('group',1 )
    pads = attr.get('pads', [0, 0]*spatiall_rank)
    strides = attr.get('strides', [1]*spatiall_rank)
    output_padding = attr.get('output_padding', [0]*spatiall_rank)
    out_spatial_shape = attr.get('output_shape', None)

    batch = x.shape[0]
    in_channel = x.shape[1]
    out_channel = w.shape[1] * groups
    if len(inputs) == 2 or isinstance(inputs[2], str) and inputs[2] == '':
        b = np.zeros([out_channel], dtype=np.float32)
    else:
        b = inputs[2]
    run_time_output_shape = [0] * spatiall_rank
    is_specify_out_shape = out_spatial_shape != None

    if is_specify_out_shape:
        pads = [0]*2*spatiall_rank
        for i in range(spatiall_rank):
            total_padding = strides[i] * (spatial_shape[i] - 1) + output_padding[i] + (
                        (kernel_shape[i] - 1) * dilations[i] + 1) - out_spatial_shape[i]
            if (auto_pad != 'SAME_UPPER'):
                upper = total_padding // 2
                lower = total_padding - (total_padding // 2)
            else:
                upper = total_padding - (total_padding // 2)
                lower = total_padding // 2
            pads[i] = upper
            pads[i+spatiall_rank] = lower
            run_time_output_shape[i] = out_spatial_shape[i] + upper + lower
        run_time_output_shape = [batch, out_channel] + run_time_output_shape
    else:
        out_spatial_shape = [0]*spatiall_rank
        for i in range(spatiall_rank):
            out_spatial_shape[i] = strides[i] * (spatial_shape[i] - 1) + output_padding[i] + (
                        (kernel_shape[i] - 1) * dilations[i] + 1) - pads[i] - pads[spatiall_rank+i]
            run_time_output_shape[i] = out_spatial_shape[i] + pads[i] + pads[spatiall_rank+i]
        run_time_output_shape = [batch, out_channel] + run_time_output_shape

    perm = [0] + [i + 2 for i in range(spatiall_rank)] + [1]
    x = np.transpose(x, perm)
    perm = [i + 2 for i in range(spatiall_rank)] + [1, 0]
    w = np.transpose(w, perm)

    index = [0] + [i+2 for i in range(spatiall_rank)] + [1]
    tf_out_shape = [run_time_output_shape[i] for i in index]

    _strides = [1] + strides + [1]

    dconv_func = None
    if spatiall_rank == 1:
        dconv_func = tf.nn.conv1d_transpose
    elif spatiall_rank == 2:
        dconv_func = tf.nn.conv2d_transpose
    elif spatiall_rank == 3:
        dconv_func = tf.nn.conv3d_transpose
    else:
        al.e('Unsupport {} dims ConvTranspose'.format(spatiall_rank))

    if groups == 1:
        conv = dconv_func(x, w, tf_out_shape, strides=_strides, padding='VALID', dilations=dilations)
    else:
        input_groups = tf.split(axis=-1, num_or_size_splits=groups, value=x)
        kernel_groups = tf.split(axis=-1, num_or_size_splits=groups, value=w)
        output_groups = []
        tf_out_shape[-1] = int(tf_out_shape[-1] / groups)
        for i, k in zip(input_groups, kernel_groups):
            child = dconv_func(i, k, tf_out_shape, strides=_strides, padding='VALID', dilations=dilations)
            output_groups.append(child)
        conv = tf.concat(axis=-1, values=output_groups)

    perm = [0] + [spatiall_rank + 1] + [i + 1 for i in range(spatiall_rank)]
    conv = tf.nn.bias_add(conv, b)
    for r in range(spatiall_rank):
        if pads[r] > 0:
            slice_begin = [0]*(spatiall_rank+2)
            slice_begin[1+r] = max(0, pads[r])
            slice_size = list(conv.shape)
            slice_size[1+r] = out_spatial_shape[r]
            conv = tf.slice(conv, slice_begin, slice_size)
        elif pads[r] < 0:
            tf_pads = [(0,0)]
            for i in range(spatiall_rank):
                if i != r:
                    tf_pads.append((0,0))
                else:
                    tf_pads.append((0, abs(pads[r])))
            tf_pads.append((0, 0))

            conv = tf.pad(conv, tf_pads)

    res = tf.transpose(conv, perm)
    return res.numpy()

