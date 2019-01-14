import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule
from ..utils import xavier_init


class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class PyramidPool(nn.Module):
    def __init__(self, in_features, out_features, pool_size):
        super(PyramidPool, self).__init__()

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_features, out_features, 1, bias=True),
            #nn.BatchNorm2d(out_features, momentum=.95),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size=x.size()
        output=F.upsample(self.features(x), size[2:], mode='bilinear')
        return output


class LargeSeparableConv2d(nn.Module):
  def __init__(self, c_in, c_out, kernel_size=15, bias=False, bn=False, setting='L'):
    super(LargeSeparableConv2d, self).__init__()
    
    dim_out = 10 * 7 * 7    
    c_mid = 64 if setting == 'S' else 256

    self.din = c_in
    self.c_mid = c_mid
    self.c_out = c_out
    self.k_height = (kernel_size, 1)
    self.k_width = (1, kernel_size)
    self.pad_height = (kernel_size//2, 0)
    self.pad_width = (0, kernel_size//2)
    self.bias = bias
    self.bn = bn

    self.block1_1 = nn.Conv2d(self.din, self.c_mid, self.k_width, 1, padding=self.pad_width, bias=self.bias)
    self.bn1_1 = nn.BatchNorm2d(self.c_mid)
    self.block1_2 = nn.Conv2d(self.c_mid, self.c_out, self.k_height, 1, padding=self.pad_height, bias=self.bias)
    self.bn1_2 = nn.BatchNorm2d(self.c_out)

    self.block2_1 = nn.Conv2d(self.din, self.c_mid, self.k_height, 1, padding=self.pad_height, bias=self.bias)
    self.bn2_1 = nn.BatchNorm2d(self.c_mid)
    self.block2_2 = nn.Conv2d(self.c_mid, self.c_out, self.k_width, 1, padding=self.pad_width, bias=self.bias)
    self.bn2_2 = nn.BatchNorm2d(self.c_out)

  def forward(self, x):
    x1 = self.block1_1(x)
    x1 = self.bn1_1(x1) if self.bn else x1
    x1 = self.block1_2(x1)
    x1 = self.bn1_2(x1) if self.bn else x1

    x2 = self.block2_1(x)
    x2 = self.bn2_1(x2) if self.bn else x2
    x2 = self.block2_2(x2)
    x2 = self.bn2_2(x2) if self.bn else x2

    return x1 + x2

class GlobalPoolFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(GlobalPoolFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # global pool
        self.pool1 = PyramidPool(in_channels[-1], out_channels//4, 1)
        self.pool2 = PyramidPool(in_channels[-1], out_channels//4, 2)
        self.pool3 = PyramidPool(in_channels[-1], out_channels//4, 3)
        self.pool4 = PyramidPool(in_channels[-1], out_channels//4, 6)
        
        self.nei_conv = LargeSeparableConv2d(in_channels[-1], out_channels, bias=True)

        self.red_conv = nn.Conv2d(out_channels*3, out_channels,  1)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def _global_pool(self, x):
        return torch.cat([self.pool1(x),
                          self.pool2(x),
                          self.pool3(x),
                          self.pool4(x)], dim=1)

    def _local_global_neighboor(self, x, l_feat):
        g_feat = self._global_pool(x)
        n_feat = self.nei_conv(x)
        lgn_feat = torch.cat([l_feat, g_feat, n_feat], dim=1)
        return self.red_conv(lgn_feat)


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        lgn_feat = self._local_global_neighboor(inputs[-1], laterals[-1])
        laterals[-1] = lgn_feat

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
