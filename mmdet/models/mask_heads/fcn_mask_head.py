import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import ConvModule
from mmdet.core import mask_cross_entropy, mask_target


class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels
                           if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                self.conv_out_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
        else:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms

class MSAdaFCNMaskHead(FCNMaskHead):

    def __init__(self,
                 num_lvls=4,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 out_reso=28,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None):
        super(FCNMaskHead, self).__init__()
        
        self.conv_out_channels = conv_out_channels
        self.out_reso = out_reso
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        with_bias = normalize is None 

        self.ada_convs1 = nn.ModuleList()
        for i in range(num_lvls):
            self.ada_convs1.append(ConvModule(in_channels,
                                              conv_out_channels,
                                              3,
                                              padding=1,
                                              normalize=normalize,
                                              bias=with_bias))

        self.ada_convs2 = nn.ModuleList()
        for i in range(num_lvls):
            self.ada_convs2.append(ConvModule(in_channels,
                                              conv_out_channels,
                                              3,
                                              padding=1,
                                              normalize=normalize,
                                              bias=with_bias))

        self.ada_convs3 = nn.ModuleList()
        for i in range(num_lvls):
            self.ada_convs3.append(ConvModule(in_channels,
                                              conv_out_channels,
                                              3,
                                              padding=1,
                                              normalize=normalize,
                                              bias=with_bias))
        
        convs1 = []
        for i in range(num_convs):
            convs1.append(ConvModule(in_channels,
                                     conv_out_channels,
                                     3,
                                     padding=1,
                                     normalize=normalize,
                                     bias=with_bias))
        convs2 = []
        for i in range(num_convs):
            convs2.append(ConvModule(in_channels,
                                     conv_out_channels,
                                     3,
                                     padding=1,
                                     normalize=normalize,
                                     bias=with_bias))

        convs3 = []
        for i in range(num_convs):
            convs3.append(ConvModule(in_channels,
                                     conv_out_channels,
                                     3,
                                     padding=1,
                                     normalize=normalize,
                                     bias=with_bias))

        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(*convs2)
        self.convs3 = nn.Sequential(*convs3)

        num_classes = 1 if class_agnostic else num_classes
        self.conv_logits = nn.Conv2d(conv_out_channels, num_classes, 1)


    def init_weights(self):
        for m in [self.conv_logits]:
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, tuple)

        inds1, inds2, inds3, x1, x2, x3  = x

        def _ada_forward(ada_convs, x):
            for i in range(len(x)):
                x[i] = ada_convs[i](x[i])
            for i in range(1, len(x)):
                x[0] = torch.max(x[0], x[i])
            x = x[0]
            return x

        def _up(x):
            return interpolate(x, scale_factor=2, mode='bilinear')

        def _insert(x, indsi, xi):
            if xi.numel() > 0:
                x[indsi] = xi

        # convs1 pass
        x1 = _ada_forward(self.ada_convs1, x1)
        for conv in self.convs1:
            x1 = conv(x1)
        x1 = _up(x1)

        # convs2 pass
        x2 = _ada_forward(self.ada_convs2, x2)
        for conv in self.convs2:
            x1 = conv(x1)
            x2 = conv(x2)
        x1 = _up(x1)
        x2 = _up(x2)

        # convs3 pass
        x3 = _ada_forward(self.ada_convs3, x3)
        for conv in self.convs3:
            x1 = conv(x1)
            x2 = conv(x2)
            x3 = conv(x3)

        # merge x
        x = torch.cuda.FloatTensor(len(inds1), self.conv_out_channels,
                                   self.out_reso, self.out_reso).fill_(0)

        _insert(x, inds1, x1)
        _insert(x, inds2, x2)
        _insert(x, inds3, x3)

        mask_pred = self.conv_logits(x)
        return mask_pred

from torch.nn.modules.utils import _ntuple
import math
def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    #output_shape = tuple(_output_size(2))
    #output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, [0])

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


