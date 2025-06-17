from archs import register

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dim=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()
        assert dim in [1, 2, 3]
        self.dim = dim
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dim == 3:
            conv_nd = nn.Conv3d
            max_pool_nd = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            max_pool_nd = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_nd = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0
        )
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0
                ),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.phi = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0
        )
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_nd)
            self.phi = nn.Sequential(self.phi, max_pool_nd)

    def forward(self, x):
        b = x.size(0)

        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        z = w_y + x

        return z


class NonLocalBlock1D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock1D, self).__init__(
            in_channels, inter_channels=inter_channels, dim=1, sub_sample=sub_sample, bn_layer=bn_layer
        )


class NonLocalBlock2D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock2D, self).__init__(
            in_channels, inter_channels=inter_channels, dim=2, sub_sample=sub_sample, bn_layer=bn_layer
        )


class NonLocalBlock3D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock3D, self).__init__(
            in_channels, inter_channels=inter_channels, dim=3, sub_sample=sub_sample, bn_layer=bn_layer
        )


class NonLocal3DWrapper(nn.Module):
    def __init__(self, block, n_seg):
        super(NonLocal3DWrapper, self).__init__()
        self.block = block
        self.n_seg = n_seg
        self.nl = NonLocalBlock3D(block.bn3.num_features)

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_seg, self.n_seg, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


class TemporalShiftBlock(nn.Module):
    def __init__(self, net, n_seg=3, n_div=8, inplace=False):
        super(TemporalShiftBlock, self).__init__()
        self.net = net
        self.n_seg = n_seg
        self.n_div = n_div
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_seg, n_div=self.n_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_seg, n_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_seg
        x = x.view(n_batch, n_seg, c, h, w)

        fold = c // n_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)
