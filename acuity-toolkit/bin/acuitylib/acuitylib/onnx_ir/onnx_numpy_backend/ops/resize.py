from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from acuitylib.acuitylog import AcuityLog as al
from acuitylib.xtf import xtf as tf

def Resize(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    coordinate_transformation_mode = attr.get('coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a = attr.get('cubic_coeff_a', -0.75)
    exclude_outside = attr.get('exclude_outside', 0)
    extrapolation_value = attr.get('extrapolation_value', 0.0)
    mode = attr.get('mode', 'nearest')
    nearest_mode = attr.get('nearest_mode', 'round_prefer_floor')

    if coordinate_transformation_mode == 'tf_half_piexl_for_nn' or \
            coordinate_transformation_mode == 'tf_crop_and_resize':
        al.e("Unsuport coordinate_transformation_mode '{}'".format(coordinate_transformation_mode))

    if mode == 'nearest' and 'ceil' in nearest_mode:
        al.e("Unsupport nearest with ceil mode")

    # pytorch coeff_a is -0.75
    # tf coeff_a is -0.5, we only support this coeff_a
    if mode == 'cubic' and cubic_coeff_a != -0.5:
        al.e("Only support cubic_coeff_a is equal to -0.5")

    if op_version < 11:
        scales = inputs[1]
        roi = None
        sizes = None
    else:
        roi = None if len(inputs) < 2 or not isinstance(inputs[1], np.ndarray) or inputs[1].size == 0 else inputs[1]
        scales = None if len(inputs) < 3 or not isinstance(inputs[2], np.ndarray) or inputs[2].size == 0 else inputs[2]
        sizes = None if len(inputs) < 4 else inputs[3]

    data = inputs[0]
    # This is a correction value to fix the accuracy issue of float64 to float32
    # eg: out = 35.0, in = 18.0, scale = 35.0 / 18.0 = 1.94444444
    #     but out = floor(1.9444444 * 18.9) = 34
    if scales is not None:
        correction_value = 1e-5
        new_shape = np.floor(np.multiply(data.shape, scales) + correction_value).astype(np.int32)
    elif sizes is not None:
        new_shape = sizes.astype(np.int32)
    else:
        al.e("Only one of scales and sizes can be specified")

    if 'infer_shape' in attr:
        return np.ones(new_shape, data.dtype)

    if len(data.shape) < 2 or len(data.shape) > 4:
        al.e("Only support 3D or 4D resize")

    if data.shape[0] != new_shape[0] or data.shape[1] != new_shape[1]:
        al.e("Only support resize width or height")

    spatial_rank = data.ndim - 2
    perm = [0] + [i + 2 for i in range(spatial_rank)] + [1]
    x = np.transpose(data, perm)
    new_spatial_shape = new_shape[2:] # [batch, channel, height, width] or [batch, channel, width]

    align_corners = coordinate_transformation_mode == 'align_corners'
    half_pixel = coordinate_transformation_mode in ['half_pixel', 'pytorch_half_pixel']
    if mode == 'nearest':
        upsamp = tf.compat.v1.image.resize_nearest_neighbor(x, new_spatial_shape,
                                                            align_corners, half_pixel_centers=half_pixel)
    elif mode == 'linear':
        upsamp = tf.compat.v1.image.resize_bilinear(x, new_spatial_shape, align_corners, half_pixel_centers=half_pixel)
    else:
        upsamp = tf.compat.v1.image.resize_bicubic(x, new_spatial_shape, align_corners, half_pixel_centers=half_pixel)

    perm = [0] + [spatial_rank + 1] + [i + 1 for i in range(spatial_rank)]
    res = tf.transpose(upsamp, perm)
    return res.numpy()


def Resize_(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    def cartesian(arrays, out=None):
        # type: (List[np.ndarray], np.ndarray) -> np.ndarray
        """
        From https://stackoverflow.com/a/1235363
        Generate a cartesian product of input arrays.
        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.
        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.
        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])
        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n // arrays[0].size
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out

    def interpolate_1d_with_x(data,  # type: np.ndarray
                              scale_factor,  # type: float
                              x,  # type: float
                              get_coeffs,  # type: Callable[[float], np.ndarray]
                              roi=None,  # type: np.ndarray
                              extrapolation_value=0.0,  # type: float
                              coordinate_transformation_mode='half_pixel',  # type: Text
                              exclude_outside=False,  # type: bool
                              ):  # type: (...) -> np.ndarray
        def get_neighbor_idxes(x, n, limit):  # type: (float, int, int) -> np.ndarray
            """
            Return the n nearest indexes to x among [0, limit), prefer the indexes smaller than x.
            As a result, the ratio must be in (0, 1]
            Examples:
            get_neighbor_idxes(4, 2, 10) == [3, 4]
            get_neighbor_idxes(4, 3, 10) == [3, 4, 5]
            get_neighbor_idxes(4.4, 3, 10) == [3, 4, 5]
            get_neighbor_idxes(4.5, 3, 10) == [3, 4, 5]
            get_neighbor_idxes(4.6, 3, 10) == [4, 5, 6]
            get_neighbor_idxes(4.4, 1, 10) == [4]
            get_neighbor_idxes(4.6, 1, 10) == [5]
            :param x:
            :param n: the number of the wanted indexes
            :param limit: the maximum value of index
            :return: An np.array containing n nearest indexes in ascending order
            """
            idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
            idxes = sorted(idxes)
            return np.array(idxes)

        def get_neighbor(x, n, data):  # type: (float, int, np.ndarray) -> np.ndarray
            """
            Pad `data` in 'edge' mode, and get n nearest elements in the padded array and their indexes in the original
            array
            :param x: center index (in the unpadded coordinate system) of the found nearest elements.
            :param n: the number of neighbors.
            :param data: the array
            :return: A tuple containing the indexes of neighbor elements (the index can be smaller than 0 or higher than
            len(data)) and the value of these elements
            """
            pad_width = np.ceil(n / 2).astype(np.int)
            padded = np.pad(data, pad_width, mode='edge')
            x += pad_width

            idxes = get_neighbor_idxes(x, n, len(padded))
            ret = padded[idxes]
            return idxes - pad_width, ret

        input_width = len(data)
        output_width = scale_factor * input_width
        if coordinate_transformation_mode == 'align_corners':
            if output_width == 1:
                x_ori = 0.
            else:
                x_ori = x * (input_width - 1) / (output_width - 1)
        elif coordinate_transformation_mode == 'asymmetric':
            x_ori = x / scale_factor
        elif coordinate_transformation_mode == 'tf_crop_and_resize':
            if output_width == 1:
                x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
            else:
                x_ori = x * (roi[1] - roi[0]) * \
                        (input_width - 1) / (output_width - 1)
            x_ori += (roi[0] * (input_width - 1))
            # Return extrapolation_value directly as what TF CropAndResize does
            if x_ori < 0 or x_ori > input_width - 1:
                return extrapolation_value
        elif coordinate_transformation_mode == 'tf_half_pixel_for_nn':
            x_ori = (x + 0.5) / scale_factor
        elif coordinate_transformation_mode == 'pytorch_half_pixel':
            if output_width == 1:
                x_ori = -0.5
            else:
                x_ori = (x + 0.5) / scale_factor - 0.5
        else:  # coordinate_transformation_mode == 'half_pixel'
            x_ori = (x + 0.5) / scale_factor - 0.5
        x_ori_int = np.floor(x_ori).astype(np.int).item()

        # ratio must be in (0, 1] since we prefer the pixel on the left of `x_ori`
        if x_ori.is_integer():
            ratio = 1
        else:
            ratio = x_ori - x_ori_int

        coeffs = get_coeffs(ratio)
        n = len(coeffs)

        idxes, points = get_neighbor(x_ori, n, data)

        if exclude_outside:
            for i, idx in enumerate(idxes):
                if idx < 0 or idx >= input_width:
                    coeffs[i] = 0
            coeffs /= sum(coeffs)

        return np.dot(coeffs, points).item()

    def interpolate_nd_with_x(data,  # type: np.ndarray
                              n,  # type: int
                              scale_factors,  # type: List[float]
                              x,  # type: List[float]
                              get_coeffs,  # type: Callable[[float], np.ndarray]
                              roi=None,  # type: np.ndarray
                              **kwargs  # type: Any
                              ):  # type: (...) -> np.ndarray
        if n == 1:
            return interpolate_1d_with_x(data, scale_factors[0], x[0], get_coeffs, roi=roi,
                                         **kwargs)
        return interpolate_1d_with_x(
            [interpolate_nd_with_x(data[i], n - 1, scale_factors[1:], x[1:], get_coeffs,
                                   roi=None if roi is None else np.concatenate(
                                       [roi[1:n], roi[n + 1:]]),
                                   **kwargs)
             for i in range(data.shape[0])], scale_factors[0], x[0], get_coeffs,
            roi=None if roi is None else [roi[0], roi[n]], **kwargs)

    def interpolate_nd(data,  # type: np.ndarray
                       get_coeffs,  # type: Callable[[float], np.ndarray]
                       output_size=None,  # type: Optional[List[int]]
                       scale_factors=None,  # type: Optional[List[float]]
                       roi=None,  # type: np.ndarray
                       **kwargs  # type: Any
                       ):  # type: (...) -> np.ndarray
        def get_all_coords(data):  # type: (np.ndarray, list, dict, int) -> np.ndarray
            return cartesian([list(range(data.shape[i])) for i in range(len(data.shape))])

        assert output_size is not None or scale_factors is not None
        if output_size is not None:
            scale_factors = np.array(output_size) / np.array(data.shape)
        else:
            output_size = (scale_factors * np.array(data.shape)).astype(np.int)
        assert scale_factors is not None

        ret = np.zeros(output_size)
        for x in get_all_coords(ret):
            ret[tuple(x)] = interpolate_nd_with_x(data, len(data.shape), scale_factors, x, get_coeffs, roi=roi,
                                                  **kwargs)
        return ret

    def cubic_coeffs(ratio, A=-0.75):  # type: (float, float) -> np.ndarray
        coeffs = [((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
                  ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
                  ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
                  ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) * ((1 - ratio) + 1) - 4 * A]

        return np.array(coeffs)

    def linear_coeffs(ratio):  # type: (float) -> np.ndarray
        return np.array([1 - ratio, ratio])

    def nearest_coeffs(ratio, mode='round_prefer_floor'):  # type: (float, Text) -> np.ndarray
        if type(ratio) == int or ratio.is_integer():
            return np.array([0, 1])
        elif mode == 'round_prefer_floor':
            return np.array([ratio <= 0.5, ratio > 0.5])
        elif mode == 'round_prefer_ceil':
            return np.array([ratio < 0.5, ratio >= 0.5])
        elif mode == 'floor':
            return np.array([1, 0])
        elif mode == 'ceil':
            return np.array([0, 1])

    coordinate_transformation_mode = attr.get('coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a  = attr.get('cubic_coeff_a', -0.75)
    exclude_outside  = attr.get('exclude_outside', 0)
    extrapolation_value  = attr.get('extrapolation_value', 0.0)
    mode  = attr.get('mode', 'nearest')
    nearest_mode = attr.get('nearest_mode', 'round_prefer_floor')

    data = inputs[0]
    if op_version < 11:
        scales = inputs[1]
        roi = None
        sizes = None
    else:
        roi = None if inputs[1].size == 0 else inputs[1]
        scales = inputs[2]
        sizes = None if len(inputs) < 4 else inputs[3]

    func_schema = lambda x: nearest_coeffs(x)
    if mode == 'cubic':
        func_schema = lambda x: cubic_coeffs(x, A=cubic_coeff_a)
    elif mode == 'linear':
        func_schema = lambda x: linear_coeffs(x)
    elif mode == 'nearest':
        func_schema = lambda x: nearest_coeffs(x, mode=nearest_mode)

    if 'infer_shape' in attr:
        output_size = sizes
        scale_factors=scales
        assert output_size is not None or scale_factors is not None
        if output_size is not None:
            scale_factors = np.array(output_size) / np.array(data.shape)
        else:
            output_size = (scale_factors * np.array(data.shape)).astype(np.int)

        return np.ones(output_size, dtype=data.dtype)

    return interpolate_nd(data, func_schema, scale_factors=scales, exclude_outside=exclude_outside, roi=roi,
                          coordinate_transformation_mode=coordinate_transformation_mode, output_size=sizes,
                          extrapolation_value=extrapolation_value).astype(np.float32)

