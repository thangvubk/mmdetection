from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = torch.cuda.FloatTensor(rois.size()[0], self.out_channels,
                                           out_size, out_size).fill_(0)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
        return roi_feats


class AdaRoIExtractor(SingleRoIExtractor):
    
    def forward(self, feats, rois):
        assert len(feats) > 1
        
        roi_feats = []
        for feat, roi_layer in zip(feats, self.roi_layers):
            roi_feats.append(roi_layer(feat, rois))
        return roi_feats

class MSAggRoIExtractor(SingleRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        # TODO add roi_layer to config
        super(MSAggRoIExtractor, self).__init__(roi_layer, out_channels, featmap_strides)
        roi_layers_lvl1 = dict(type='RoIAlign', out_size=7, sample_num=2)
        roi_layers_lvl2 = dict(type='RoIAlign', out_size=14, sample_num=2)
        roi_layers_lvl3 = dict(type='RoIAlign', out_size=28, sample_num=2)
        self.roi_layers_lvl1 = self.build_roi_layers(roi_layers_lvl1, featmap_strides)
        self.roi_layers_lvl2 = self.build_roi_layers(roi_layers_lvl2, featmap_strides)
        self.roi_layers_lvl3 = self.build_roi_layers(roi_layers_lvl3, featmap_strides)

    def forward(self, feats, rois):
        assert len(feats) > 1

        roi_feats_lvl1 = []
        for feat, roi_layer in zip(feats, self.roi_layers_lvl1):
            roi_feats_lvl1.append(roi_layer(feat, rois))

        roi_feats_lvl2 = []
        for feat, roi_layer in zip(feats, self.roi_layers_lvl2):
            roi_feats_lvl2.append(roi_layer(feat, rois))

        roi_feats_lvl3 = []
        for feat, roi_layer in zip(feats, self.roi_layers_lvl3):
            roi_feats_lvl3.append(roi_layer(feat, rois))
        return roi_feats_lvl1, roi_feats_lvl2, roi_feats_lvl3

class MSAdaRoIExtractor(SingleRoIExtractor):
    def __init__(self,
                 roi_layer,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        #import pdb; pdb.set_trace()
        self.roi_layers_1 = self.build_roi_layers(roi_layer[0], featmap_strides)
        self.roi_layers_2 = self.build_roi_layers(roi_layer[1], featmap_strides)
        self.roi_layers_3 = self.build_roi_layers(roi_layer[2], featmap_strides)

    def inds_from_lvls(self, lvls, set_id):
        """
        lvl0 + lvl1 => roi set 1
        lvl2        => roi set 2
        lvl3        => roi set 3
        """
        if set_id == 1:
            inds = (lvls == 0)
            inds += (lvls == 1)
        else:
            inds = (lvls == set_id)
        return inds


    def forward(self, feats, rois):
        assert len(feats) > 1

        def _get_roi_feats(roi_layers, inds):
            # TODO handle empty rois
            # assert inds.any()
            rois_ = rois[inds, :]
            roi_feats = []
            for feat, roi_layer in zip(feats, roi_layers):
                if rois_.numel() > 0:
                    roi_feats.append(roi_layer(feat, rois_))
                else:
                    shape = [0, feat.size(1), roi_layer.out_size, roi_layer.out_size]
                    roi_feats.append(_NewEmptyTensorOp.apply(feat, shape))
            return roi_feats

        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        inds_1 = self.inds_from_lvls(target_lvls, 1)
        inds_2 = self.inds_from_lvls(target_lvls, 2)
        inds_3 = self.inds_from_lvls(target_lvls, 3)

        roi_feats_1 = _get_roi_feats(self.roi_layers_1, inds_1)
        roi_feats_2 = _get_roi_feats(self.roi_layers_2, inds_2)
        roi_feats_3 = _get_roi_feats(self.roi_layers_3, inds_3)

        return (inds_1, inds_2, inds_3, 
                roi_feats_1, roi_feats_2, roi_feats_3)

        
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

