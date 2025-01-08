from acuitylib.layer.customlayer import CustomLayer
from acuitylib.core.shape import Shape
import numpy as np
from acuitylib.xtf import xtf as tf

class CollectAndDistributeFpnRpnProposals(CustomLayer):

    op = 'collect_and_distribute_fpn_rpn_proposals'

    def _py_collect(self, rois, scores):
        p = self.params
        scores = scores.squeeze()
        inds = np.argsort(-scores)[:p.rpn_post_nms_top_n].reshape([-1])
        rois = rois[inds, :]
        return rois

    def _py_distribute(self, rois):
        p = self.params
        outputs = []
        def boxes_area(boxes):
            w = (boxes[:, 2] - boxes[:, 0] + 1)
            h = (boxes[:, 3] - boxes[:, 1] + 1)
            areas = w * h
            assert np.all(areas >= 0), 'Negative areas founds'
            return areas
        def map_rois_to_fpn_levels(rois, k_min, k_max):
            s = np.sqrt(boxes_area(rois))
            s0 = p.roi_canonical_scale
            lvl0 = p.roi_canonical_level

            target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
            target_lvls = np.clip(target_lvls, k_min, k_max)
            return target_lvls
        lvl_min = p.roi_min_level
        lvl_max = p.roi_max_level
        lvls = map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
        outputs.append(rois)

        rois_idx_order = np.empty((0, ))
        for lvl in range(lvl_min, lvl_max + 1):
            idx_lvl = np.where(lvls == lvl)[0]
            if idx_lvl.size > 0:
                blob_roi_level = rois[idx_lvl, :]
            outputs.append(blob_roi_level)
            rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_idx_restore = np.argsort(rois_idx_order)
        outputs.append(rois_idx_restore.astype(np.int32))
        return outputs

    def setup(self, inputs, outputs):
        p = dict()
        p['rpn_post_nms_top_n'] = 1000
        p['rpn_max_level'] = 6
        p['rpn_min_level'] = 2
        p['roi_max_level'] = 5
        p['roi_min_level'] = 2
        p['roi_canonical_scale'] = 224
        p['roi_canonical_level'] = 4
        self.set_params(p)
        p = self.params

        for i in range(1, int((len(inputs) + 1)/2)):
            outputs[i].shape = Shape([0,5])
        outputs[0].shape = Shape([p.rpn_post_nms_top_n,5])
        outputs[-1].shape = Shape([p.rpn_post_nms_top_n, 1])

    def compute_out_tensor(self, tensor, input_tensor):
        ''' Only for inference
        '''
        p = self.params
        # collect
        post_nms_top_n = p.rpn_post_nms_top_n
        k_max = p.rpn_max_level
        k_min = p.rpn_min_level
        num_lvls = k_max - k_min + 1
        roi_inputs = input_tensor[:num_lvls]
        score_inputs = input_tensor[num_lvls:]

        rois = tf.concat(roi_inputs, axis = 0)
        scores = tf.concat(score_inputs, axis = 0)
        rois = tf.numpy_function(self._py_collect, [rois, scores], [tf.float32])

        # distribute
        lvl_min = p.roi_min_level
        lvl_max = p.roi_max_level
        outs = tf.numpy_function(self._py_distribute, rois, [tf.float32]*(lvl_max - lvl_min + 2) + [tf.int32])
        for out in outs[1:-1]:
            out.set_shape([None, 5])
        outs[0].set_shape([post_nms_top_n, 5])
        outs[-1].set_shape([input_tensor[0].get_shape().dims[0], 1])

        return outs


