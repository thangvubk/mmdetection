import torch
import torch.nn as nn
import torch.nn.functional as F

from .bbox_head import BBoxHead
from ..utils import ConvModule


class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 normalize=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs
                + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalize = normalize
        self.with_bias = normalize is None

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else
                           4 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (last_layer_dim
                                    if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        normalize=self.normalize,
                        bias=self.with_bias))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim
                                  if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

class MSAdaConvFCBBoxHead(ConvFCBBoxHead, BBoxHead):

    def __init__(self,
                 num_shared_convs=4,
                 num_lvls=4,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 normalize=None,
                 *args,
                 **kwargs):

        assert num_shared_convs >= 1 and num_lvls > 1
        BBoxHead.__init__(self, *args, **kwargs)

        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalize = normalize
        self.with_bias = normalize is None

        self.ada_convs1 = nn.ModuleList()
        for i in range(num_lvls):
            self.ada_convs1.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.conv_out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)))

        self.ada_convs2 = nn.ModuleList()
        for i in range(num_lvls):
            self.ada_convs2.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.conv_out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)))

        self.ada_convs3 = nn.ModuleList()
        for i in range(num_lvls):
            self.ada_convs3.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.conv_out_channels, 3, 1, 1),
                nn.ReLU(inplace=True)))

        self.convs3, _, _ = \
            self._add_conv_fc_branch(
                2, 0, self.conv_out_channels,
                True)

        self.convs2, _, _ = \
            self._add_conv_fc_branch(
                2, 0, self.conv_out_channels*2,
                True)

        self.convs1, self.fc, last_dim = \
            self._add_conv_fc_branch(
                2, 1, self.conv_out_channels*2,
                True)

        if self.with_cls:
            self.fc_cls = nn.Linear(last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else
                           4 * self.num_classes)
            self.fc_reg = nn.Linear(last_dim, out_dim_reg)

    def init_weights(self):
        BBoxHead.init_weights(self)
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, 0)

    def forward(self, x):

        # adaptive feature pooling
        def _ada_forward(ada_convs, x):
            for i in range(len(x)):
                x[i] = ada_convs[i](x[i])
            for i in range(1, len(x)):
                x[0] = torch.max(x[0], x[i])
            x = x[0]
            return x

        def _down(x):
            return F.avg_pool2d(x, 2)

        assert isinstance(x, tuple) and isinstance(x[0], list)
        x1, x2, x3 = x
        x1 = _ada_forward(self.ada_convs1, x1)
        x2 = _ada_forward(self.ada_convs2, x2)
        x3 = _ada_forward(self.ada_convs3, x3)

        for conv in self.convs3:
            x3 = conv(x3)

        x2 = torch.cat([x2, _down(x3)], dim=1)
        for conv in self.convs2:
            x2 = conv(x2)

        x1 = torch.cat([x1, _down(x2)], dim=1)
        for conv in self.convs1:
            x1 = conv(x1)
        x1 = x1.view(x1.size(0), -1)

        x1 = self.fc[0](x1)

        cls_score = self.fc_cls(x1) if self.with_cls else None
        bbox_pred = self.fc_reg(x1) if self.with_reg else None
        return cls_score, bbox_pred
