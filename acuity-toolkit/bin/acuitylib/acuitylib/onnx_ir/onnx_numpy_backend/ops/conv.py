from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import copy
from acuitylib.acuitylog import AcuityLog as al

def Conv(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    from .max_pool import get_output_shape, get_pad_shape
    x = copy.copy(inputs[0])
    x_shape = np.shape(x)
    spatial_rank = x.ndim - 2
    w = inputs[1]

    # auto_pad = attr.get('auto_pad', 'NOTSET')
    # kernel_shape = attr.get('kernel_shape', w.shape[2:])
    if len(inputs) == 2 or isinstance(inputs[2], str) and inputs[2] == '':
        b = np.zeros([w.shape[0]], dtype=x.dtype)
    else:
        b = inputs[2]
    dilations = attr.get('dilations', [1] * spatial_rank)
    groups = attr.get('group',1 )
    pads = attr.get('pads', [0, 0]*spatial_rank)
    strides = attr.get('strides', [1]*spatial_rank)
    ceil_mode = attr.get('ceil_mode', 0) == 1
    auto_pad = attr.get('auto_pad', 'VALID')

    if 'kernel_shape' not in attr:
        kernel_shape = [1] * spatial_rank
        if spatial_rank == 2:
            kernel_shape[0] = w.shape[2]
            kernel_shape[1] = w.shape[3]
        elif spatial_rank == 1:
            kernel_shape[0] = w.shape[2]
    else:
        kernel_shape = attr['kernel_shape']

    out_spatial_shape = get_output_shape(auto_pad, x_shape[2:], kernel_shape, strides,
                                         dilations=dilations, ceil_mode=ceil_mode, spatial_pad=pads)
    spatial_pads = get_pad_shape(auto_pad, x_shape[2:], kernel_shape, strides, dilations,
                                 out_spatial_shape, pads)
    spatial_pads = np.array(spatial_pads)
    for idx in range(len(spatial_pads)):
        pads[idx] = spatial_pads[idx][0]
        pads[idx + spatial_rank] = spatial_pads[idx][1]

    import torch
    import torch.nn.functional as F
    conv_func = None
    if spatial_rank == 1:
        conv_func = F.conv1d
    elif spatial_rank == 2:
        conv_func = F.conv2d
    elif spatial_rank == 3:
        conv_func = F.conv3d
    else:
        al.e("Not support {} dim conv".format(str(spatial_rank)))

    asymmetric_pad = False
    for s in range(spatial_rank):
        if pads[s] != pads[s +spatial_rank]:
            asymmetric_pad = True
    torch_pad = list()
    if asymmetric_pad:
        pre_pad_for_asymmetric_pad = [[0,0],[0,0]]
        for s in range(spatial_rank):
            pad_begin = 0
            pad_end = 0
            if pads[s] == pads[s + spatial_rank]:
                torch_pad.append(pads[s*2])
            elif pads[s] < pads[s + spatial_rank]:
                torch_pad.append(pads[s])
                pad_end = pads[s + spatial_rank] - pads[s]
            else:
                torch_pad.append(pads[s + spatial_rank])
                pad_begin = pads[s] - pads[s + spatial_rank]
            pre_pad_for_asymmetric_pad.append([pad_begin, pad_end])
        x = np.pad(x, pre_pad_for_asymmetric_pad, 'constant')
    else:
        for s in range(spatial_rank):
            torch_pad.append(pads[s])

    in_torch_tensor = torch.tensor(x.astype(np.float32), requires_grad=False)
    w_torch_tensor = torch.tensor(w.astype(np.float32), requires_grad=False)
    b_torch_tensor = torch.tensor(b.astype(np.float32), requires_grad=False)

    with torch.no_grad():
        res_torch_tensor = conv_func(in_torch_tensor,
                                     w_torch_tensor,
                                     b_torch_tensor,
                                     strides,
                                     torch_pad,
                                     dilations,
                                     groups
                                     )

    res = res_torch_tensor.detach().numpy()
    return res
