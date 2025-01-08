from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import copy
from acuitylib.acuitylog import AcuityLog as al

def MaxUnpool(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    X = inputs[0]
    I = inputs[1]
    output_shape = None if len(inputs) == 2 else inputs[2]
    spatial_rank = X.ndim - 2

    kernel_shape = attr.get('kernel_shape')
    pads = attr.get('pads', [0, 0]*spatial_rank)
    strides = attr.get('strides', [1]*spatial_rank)

    import torch
    import torch.nn.functional as F
    unpool_func = None
    if spatial_rank == 1:
        unpool_func = F.max_unpool1d
    elif spatial_rank == 2:
        unpool_func = F.max_unpool2d
    elif spatial_rank == 3:
        unpool_func = F.max_unpool3d
    else:
        al.e("Not support {} dim max_unpool".format(str(spatial_rank)))

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
        X = np.pad(X, pre_pad_for_asymmetric_pad, 'constant')
    else:
        for s in range(spatial_rank):
            torch_pad.append(pads[s])

    input_torch_tensor = torch.tensor(X, requires_grad=False)
    # Index use default size to calc, the output will pad zeros
    index_torch_tensor = torch.tensor(I, requires_grad=False)
    with torch.no_grad():
        res_torch_tensor = unpool_func(input_torch_tensor,
                                     index_torch_tensor,
                                     kernel_shape,
                                     strides,
                                     torch_pad,
                                     None
                                     )

    res = res_torch_tensor.detach().numpy()
    if isinstance(output_shape, np.ndarray):
        need_pad = res.shape != tuple(output_shape)
        if need_pad == True:
            need_pad_size = output_shape - res.shape
            pads = [(0, p) for p in need_pad_size]
            res = np.pad(res, pads)
    return res
