#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06

import torch.nn as nn
from .DnCNN import DnCNN,DnCNN_R
from .UNet import UNet
# from.NestedUNet import UNet  #mcj
# from .NestedUNet import NestedUNet
# from .NestedUNet import UNet
# from .NestedUNet_V1 import UNet
# from .NestedUNet_V1 import NestedUNet
from .NestedUNet_V2 import NestedUNet, NestedUNet_4

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        # #VDN Unet
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)

        # NestedUNet_V2.py Unet++:NestedUNet_4 四层
        # self.DNet = NestedUNet_4(input_channels=1, out_channels=2, slope=0.2)


        # self.DNet = DnCNN_R(in_channels, in_channels*2, dep=17, num_filters=64, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma

class VDN_MSE(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN_MSE, self).__init__()
        self.DNet = DnCNN(in_channels, in_channels, dep=17, num_filters=64, slope=slope)
    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z

class VDN_sigma(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN_sigma, self).__init__()
        # self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        self.DNet = DnCNN(in_channels, in_channels, dep=17, num_filters=64, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z

class VDN_vi(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN_vi, self).__init__()
        # self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        self.DNet = DnCNN(in_channels, in_channels * 2, dep=dep_S, num_filters=64, slope=slope)
    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
