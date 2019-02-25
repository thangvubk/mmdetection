import torch.nn as nn
from mmcv.cnn import constant_init

from .norm import build_norm_layer


class ResBlock(nn.Module):
    """ ResBlock for bbox head and mask head """

    def __init__(self, in_channels, out_channels, normalize):
        super(ResBlock, self).__init__()
        with_bias = True if normalize is None else False

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=with_bias)
        self.normalize = normalize
        if normalize is not None:
            self.norm_name, norm = build_norm_layer(normalize, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (nn.Conv2d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)

        self.init_weights()

    def init_weights(self):
        if self.normalize is not None:
            constant_init(self.norm, 0)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.normalize is not None:
            out = self.norm(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
