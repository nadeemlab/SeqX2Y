from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# some replicate units for u-net
class ConvBNReLU(nn.Module):
    """
    combination of [conv] + [BN] + [ReLU]
    """

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvBNReLU, self).__init__()
        if isBN:
            self.convbnrelu = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.convbnrelu = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.convbnrelu(x)


class ConvNoPool(nn.Module):
    """
    conv twice and no pooling
    """

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvNoPool, self).__init__()
        self.convnopool = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x):
        return self.convnopool(x)


class ConvPool(nn.Module):
    """
    conv twice with a pooling layer follows
    """

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvPool, self).__init__()
        self.convpool = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x):
        return self.convpool(x)


class UpsampleConv(nn.Module):
    """
    upsample feature maps to given shape and conv twice (with skip connection)
    """

    def __init__(self, in_ch, out_ch, isDeconv=True, isBN=True):
        super(UpsampleConv, self).__init__()
        if isDeconv:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear3d(scale_factor=2),
                nn.Conv3d(in_ch, out_ch, kernel_size=1)
            )
        self.convtwice = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x1, x2):
        # this forward func is from (to solve the size incompatibility issue) :
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        x1 = self.up(x1)
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        # print(x1.size(), x2.size())
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # print(x1.size(), x2.size())
        x = torch.cat([x2, x1], dim=1)
        x = self.convtwice(x)
        return x


class ConvOut(nn.Module):
    """
    last layer for generating probability map
    """

    def __init__(self, in_ch):
        super(ConvOut, self).__init__()
        self.convout = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1),
      #      nn.Conv3d(in_ch, in_ch, kernel_size=1),
            nn.Conv3d(in_ch, 3, kernel_size=1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.size())
        output_ = self.convout(x)
        # print(output_.size())
        return output_