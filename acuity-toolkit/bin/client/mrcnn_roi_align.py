from acuitylib.layer.customlayer import CustomLayer
from acuitylib.core.shape import Shape
import tensorflow as tf
from acuitylib.layer.acuitylayer import IoMap
from acuitylib.layer.layer_params import DefParam

class MrcnnROIAlign(CustomLayer):

    op = 'mrcnn_roi_align'

    # label, description
    def_input = [IoMap('in0', 'in', 'input port'),
                 IoMap('in1', 'in', 'input port'),
                 IoMap('in2', 'in', 'input port'),
                 IoMap('in3', 'in', 'input port'),
                 IoMap('in4', 'in', 'input port'),
                 IoMap('in5', 'in', 'input port')]
    def_output = [IoMap('out0', 'out', 'output port')]

    def_param = [
        DefParam('proposal_count', 1000, False),
        DefParam('rpn_nms_threshold', 0.7, False),
        DefParam('pre_nms_limit', 6000, False),
        DefParam('rpn_bbox_std_dev', [0.1, 0.1, 0.2, 0.2], False),
        DefParam('pool_size', [], False),
    ]

    def setup(self, inputs, outputs):
        p = self.params
        in_shape = inputs[3].shape.dims
        outputs[0].shape = Shape([1, inputs[0].shape.dims[1],
                                  p.pool_size[0], p.pool_size[1], in_shape[3]])

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

    def log2_graph(self, x):
        return tf.math.log(x) / tf.math.log(2.0)


    def cal_align(self, boxes, image_meta,  p2, p3, p4, p5):
        p = self.params

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = self.parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        pool_shape = tuple(p.pool_size)

        feature_maps = []
        feature_maps.append(p2)
        feature_maps.append(p3)
        feature_maps.append(p4)
        feature_maps.append(p5)
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes,
                                                   box_indices, pool_shape, method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        sess = tf.compat.v1.Session()
        pooled = pooled.eval(session=sess)
        return pooled

    def compute_out_tensor(self, tensor, input_tensor):
        p = self.params
        rois = tf.reshape(input_tensor[0], [1,-1,4])
        image_meta = input_tensor[1]
        p2 = input_tensor[2]
        p3 = input_tensor[3]
        p4 = input_tensor[4]
        p5 = input_tensor[5]
        output = tf.numpy_function(self.cal_align, [rois, image_meta, p2, p3, p4, p5], tf.float32)
        output.set_shape([1, rois.shape[1], p.pool_size[0], p.pool_size[1], 256])

        return [output]

