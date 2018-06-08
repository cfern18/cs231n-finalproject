#!/usr/bin/python
# full assembly of the sub-parts to form the complete net
# Adapted from open-source implementation at https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# python 3 confusing imports :(
from .unet_parts import *
from .eval_functions import *

class UNet(nn.Module):

    ## Params needs to have fields "in channels", "out channels"
    def __init__(self, params):
        super(UNet, self).__init__()
        self.inc = inconv(params.in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, params.out_channels) #outconv includes sigmoid activation to squeeze input btw 0 and 1

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def get_loss_fn(self):
        return self.loss_fn
