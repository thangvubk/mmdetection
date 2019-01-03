import warnings

import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init

from .norm import build_norm_layer

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 normalize=None,
                 activation='relu',
                 inplace=True,
                 activate_last=True):
        super(ConvModule, self).__init__()
        self.with_norm = normalize is not None
        self.with_activatation = activation is not None
        self.with_bias = bias
        self.activation = activation
        self.activate_last = activate_last

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm = build_norm_layer(normalize, norm_channels)

        if self.with_activatation:
            assert activation in ['relu'], 'Only ReLU supported.'
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        # Default using msra init
        self.init_weights()

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if x.numel() > 0:
            if self.activate_last:
                x = self.conv(x)
                if norm and self.with_norm:
                    x = self.norm(x)
                if activate and self.with_activatation:
                    x = self.activate(x)
            else:
                if norm and self.with_norm:
                    x = self.norm(x)
                if activate and self.with_activatation:
                    x = self.activate(x)
                x = self.conv(x)
            return x
        else:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // d + 1
                for i, p, di, k, d in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

