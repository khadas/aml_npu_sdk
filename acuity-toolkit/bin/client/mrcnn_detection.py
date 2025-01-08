from acuitylib.layer.customlayer import CustomLayer
from acuitylib.core.shape import Shape
import tensorflow as tf
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.layer.layer_params import DefParam

class MrcnnDetection(CustomLayer):
    op = 'mrcnn_detection'

    # label, description
    def_input = [IoMap('in0', 'in', 'input port'),
                 IoMap('in1', 'in', 'input port'),
                 IoMap('in2', 'in', 'input port'),
                 IoMap('in3', 'in', 'input port'),
                 ]
    def_output = [IoMap('out0', 'out', 'output port')]

    def_param = [
        DefParam('bbox_std_dev', [0.1, 0.1, 0.2, 0.2], False),
        DefParam('detection_min_confidence', 0.7, False),
        DefParam('detection_max_instances', 100, False),
        DefParam('detection_nms_threshold', 0.3, False),
    ]

    def setup(self, inputs, outputs):
        p = self.params
        outputs[0].shape = Shape([1, p.detection_max_instances, 6])

    def parse_image_meta_graph(self, meta):
        image_id = meta[:, 0]
        original_image_shape = meta[:, 1:4]
        image_shape = meta[:, 4:7]
        window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
        scale = meta[:, 11]
        active_class_ids = meta[:, 12:]
        return {
            "image_id": image_id,
            "original_image_shape": original_image_shape,
            "image_shape": image_shape,
            "window": window,
            "scale": scale,
            "active_class_ids": active_class_ids,
        }

    def norm_boxes_graph(self, boxes, shape):
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        return tf.divide(boxes - shift, scale)

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

    def refine_detections_graph(self, rois, probs, deltas, window):
        p = self.params
        # Class IDs per ROI
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = self.apply_box_deltas_graph(rois, deltas_specific * p.bbox_std_dev)
        # Clip boxes to image window
        refined_rois = self.clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if p.detection_min_confidence:
            conf_keep = tf.where(class_scores >= p.detection_min_confidence)[:, 0]
            keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse_tensor_to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=p.detection_max_instances,
                iou_threshold=p.detection_nms_threshold)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = p.detection_max_instances - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep.set_shape([p.detection_max_instances])
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]
        # Keep top detections
        roi_count = p.detection_max_instances
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)] Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = p.detection_max_instances - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
        return detections

    def cal_detection(self, rois, mrcnn_class, mrcnn_bbox, image_meta):
        m = self.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        # Converts boxes from pixel coordinates to normalized coordinates.
        window = self.norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = self.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: self.refine_detections_graph(x, y, w, z),
            rois.shape[0])

        sess = tf.compat.v1.Session()
        detections_batch = detections_batch.eval(session=sess)
        return detections_batch

    def compute_out_tensor(self, tensor, input_tensor):
        p = self.params

        image_meta = input_tensor[0]
        rois = input_tensor[1]
        mrcnn_class = input_tensor[2]
        mrcnn_bbox = input_tensor[3]
        output = tf.numpy_function(self.cal_detection, [rois, mrcnn_class, mrcnn_bbox, image_meta], tf.float32)

        output.set_shape([1, p.detection_max_instances, 6])

        return [output]