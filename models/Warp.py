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



class Warp(nn.Module):
    """
    definition of U-net for segmentation
    """
    def __init__(self, size1, size2, size3):
        super(Warp, self).__init__()

        self.transformer = SpatialTransformer((size1,size2,size3))

    def forward(self, bat_img,dvf):
        #print(bat_img.shape, bat_label.shape)
        warped = self.transformer(bat_img, dvf)

        return warped


