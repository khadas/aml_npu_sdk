from acuitylib.layer.customlayer import CustomLayer
from acuitylib.core.shape import Shape
import numpy as np
import tensorflow as tf
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.layer.layer_params import DefParam

class MrcnnProposal(CustomLayer):

    op = 'mrcnn_proposal'

    # label, description
    def_input = [IoMap('in0', 'in', 'input port'),
                 IoMap('in1', 'in', 'input port'),
                 IoMap('in2', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def_param = [
        DefParam('proposal_count', 1000, False),
        DefParam('rpn_nms_threshold', 0.7, False),
        DefParam('pre_nms_limit', 6000, False),
        DefParam('rpn_bbox_std_dev', [0.1, 0.1, 0.2, 0.2], False),
    ]

    def setup(self, inputs, outputs):
        outputs[0].shape = Shape([1, self.params.proposal_count, 4])

    def batch_slice(self, inputs, graph_fn, batch_size, names=None):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = []
        for i in range(batch_size):
            inputs_slice = [x[i] for x in inputs]
            output_slice = graph_fn(*inputs_slice)
            if not isinstance(output_slice, (tuple, list)):
                output_slice = [output_slice]
            outputs.append(output_slice)
        outputs = list(zip(*outputs))
        if names is None:
            names = [None] * len(outputs)
        result = [tf.stack(o, axis=0, name=n)
                  for o, n in zip(outputs, names)]
        if len(result) == 1:
            result = result[0]
        return result

    def apply_box_deltas_graph(self, boxes, deltas):
        boxes = tf.cast(boxes, tf.float32)
        deltas = tf.cast(deltas, tf.float32)
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= tf.exp(deltas[:, 2])
        width *= tf.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
        return result

    def clip_boxes_graph(self, boxes, window):
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
        clipped.set_shape((clipped.shape[0], 4))
        return clipped

    def cal_roi(self, rpn_class, rpn_bbox, anchors):
        p = self.params
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_class[:, :, 1]
        # Box deltas [batch, num_rois, 4]
        rpn_bbx_std_dev = np.array(p.rpn_bbox_std_dev)
        deltas = rpn_bbox * np.reshape(rpn_bbx_std_dev, [1, 1, 4])
        # Anchors
        anchors = anchors

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(p.pre_nms_limit, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        sess = tf.compat.v1.Session()
        pre_nms_limit = pre_nms_limit.eval(session=sess)
        ix = ix.eval(session=sess)

        scores = self.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                  ix.shape[0])
        deltas = self.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                  ix.shape[0])
        pre_nms_anchors = self.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), ix.shape[0], names=[
            "pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        boxes = self.batch_slice([pre_nms_anchors, deltas],
                                 lambda x, y: self.apply_box_deltas_graph(x, y),
                                 ix.shape[0],
                                 names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = self.batch_slice(boxes,
                                 lambda x: self.clip_boxes_graph(x, window),
                                 ix.shape[0],
                                 names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, p.proposal_count, p.rpn_nms_threshold,
                                                   name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(p.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = self.batch_slice([boxes, scores], nms, boxes.shape[0])
        sess = tf.compat.v1.Session()
        proposals = proposals.eval(session=sess)
        return proposals

    def compute_out_tensor(self, tensor, input_tensor):
        rpn_class = input_tensor[0]
        rpn_bbox = input_tensor[1]
        anchors = input_tensor[2]
        p = self.params

        output = tf.numpy_function(self.cal_roi, [rpn_class, rpn_bbox, anchors], tf.float32)
        output.set_shape([1, p.proposal_count, 4])
        return [output]
