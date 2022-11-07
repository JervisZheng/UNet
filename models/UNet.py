import torch
from torch import nn
from models.common import *


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.in_layer = DoubleConv(3, 64)
        self.en_layer1 = EncodeBlock(64, 128)
        self.en_layer2 = EncodeBlock(128, 256)
        self.en_layer3 = EncodeBlock(256, 512)
        self.en_layer4 = EncodeBlock(512, 1024)

        self.mid_layer = UpSampling(1024, 512)

        self.de_layer4 = DecodeBlock(1024, 512)
        self.de_layer3 = DecodeBlock(512, 256)
        self.de_layer2 = DecodeBlock(256, 128)
        self.de_layer1 = DoubleConv(128, 64)
        self.out_layer = ConvBNAct(64, 3, 1, activation=False)

    def forward(self, x):
        lout1 = self.in_layer(x)
        lout2 = self.en_layer1(lout1)
        lout3 = self.en_layer2(lout2)
        lout4 = self.en_layer3(lout3)
        out = self.en_layer4(lout4)

        rout4 = self.mid_layer(out)

        rin4 = torch.cat([rout4, lout4], dim=1)
        rout3 = self.de_layer4(rin4)
        rin3 = torch.cat([rout3, lout3], dim=1)
        rout2 = self.de_layer3(rin3)
        rin2 = torch.cat([rout2, lout2], dim=1)
        rout1 = self.de_layer2(rin2)
        rin1 = torch.cat([rout1, lout1], dim=1)

        rout0 = self.de_layer1(rin1)
        return self.out_layer(rout0)


if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256)
    model = UNet()
    model.parameters()
    out = model(input)
    print(out.shape)






