import torch
from torch import nn


class ConvBNAct(nn.Module):
    """标准卷积模块"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups: int = 1,
                 bn: bool = True,
                 activation=nn.ReLU,
                 bias: bool = False
                 ):
        super(ConvBNAct, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            activation() if isinstance(activation, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)


class DoubleConv(nn.Module):
    """双层卷积模块"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=3,
                 padding=1):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size, stride=1, padding=padding, groups=1),
            ConvBNAct(out_channels, out_channels, kernel_size, stride=1, padding=padding, groups=1)
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channels: int,
                 kernel_size=1,
                 stride=1,
                 mode='nearest'):
        super(UpSampling, self).__init__()
        self.layer = nn.Sequential(
            ConvBNAct(in_channel, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.Upsample(scale_factor=2, mode=mode)
        )

    def forward(self, x):
        return self.layer(x)


class EncodeBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,):
        super(EncodeBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class DecodeBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=3):
        super(DecodeBlock, self).__init__()
        self.layer = nn.Sequential(
            DoubleConv(in_channels, out_channels, kernel_size),
            UpSampling(out_channels, out_channels // 2)
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    input = torch.randn(1, 1, 572, 572)
    print(input.shape)
    net = ConvBNAct(1, 64, 3, 1, 1)
    out = net(input)
    print(out.shape)
    net2 = EncodeBlock(64, 128)
    out = net2(out)
    print(out.shape)


