from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from acuitylib.xtf import xtf as tf

def NonMaxSuppression(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    center_point_box = attr.get('center_point_box', 0)
    boxes = inputs[0]
    scores = inputs[1]
    input_length = len(inputs)
    max_output_boxes_per_class = 0 if input_length < 3 else inputs[2]
    iou_threshold = 0.0 if input_length < 4 else inputs[3]
    score_threshold = 0.0 if input_length < 5 else inputs[4]
    nms = tf.image.non_max_suppression(np.squeeze(boxes), np.squeeze(scores), int(max_output_boxes_per_class),
                                       float(iou_threshold), float(score_threshold))
    res = nms.numpy().astype(np.int64)
    res = np.expand_dims(res, -1)
    res = np.pad(res, [[0,0],[2,0]])
    return res
