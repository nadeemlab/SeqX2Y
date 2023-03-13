from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
from layers import SpatialTransformer
from models.unet_utils import *



class Unet(nn.Module):
    """
    definition of U-net for segmentation
    """


    def __init__(self, img_ch, size1, size2, size3, fch_base=4, isBN=True, isDeconv=True, future_step=3):
        super(Unet, self).__init__()


        self.down1 = ConvNoPool(img_ch, fch_base, isBN)
        self.down2 = ConvPool(fch_base, fch_base * 2, isBN)
        self.down3 = ConvPool(fch_base * 2, fch_base * 4, isBN) #32
        #self.down4 = ConvPool(fch_base * 4, fch_base * 8, isBN)

        self.encoder = ConvPool(fch_base * 4, fch_base * 8, isBN) #64

        #self.up1 = UpsampleConv(fch_base * 16, fch_base * 8, isDeconv, isBN)
        self.up2 = UpsampleConv(fch_base * 8, fch_base * 4, isDeconv, isBN) #32
        self.up3 = UpsampleConv(fch_base * 4, fch_base * 2, isDeconv, isBN)
        self.up4 = UpsampleConv(fch_base * 2, fch_base, isDeconv, isBN)

        self.out = ConvOut(fch_base)
        self.transformer = SpatialTransformer((size1, size2, size3))
        # transformer
    def forward(self, bat_img, bat_label, future_step):
        # print(bat_img.shape, bat_pred.shape)
        # bat_pred: 1,1,3,64,64,64
        # bat_img: 1,4,2,64,64,64
        # bat_label: 2,3,64,64,64
        B, T, C, D, W, H = bat_img.size()
        DVF = []
        #print(bat_img.shape, bat_label.shape)
        for t in range(future_step):

            input_ = torch.cat([torch.reshape(bat_img[:, 0, 0, ...], [B,1,D,W,H]),
                                torch.reshape(bat_label[:,0, t, ...], [B,1,D,W,H])], 1)  # [B, t, C, D, W, H]

            d1 = self.down1(input_)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            #d4 = self.down4(d3)
            enc = self.encoder(d3)
            #u1 = self.up1(enc, d4)
            u2 = self.up2(enc, d3)
            u3 = self.up3(u2, d2)
            u4 = self.up4(u3, d1)
            dvf = self.out(u4) # out channel 3
            warped_img = self.transformer(torch.reshape(bat_img[:,0,0,...], [B,1,D,W,H]),dvf)
            #warped_ctr = self.transformer(torch.reshape(bat_img[:,0,1,...], [B,1,D,W,H]),dvf)
            #out = torch.cat([warped_img, warped_ctr],1)
            out = warped_img.unsqueeze(2) # [B 1 C D W H]
            #print(out.shape)
            if t==0:
                out_ = out
            else:
                out_ = torch.cat([out_,out],2)
            DVF += [dvf]

        DVF = torch.stack(DVF,1)
       #print(DVF.shape, out_.shape)
        return out_, DVF


