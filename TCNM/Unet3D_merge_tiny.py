"""TCNM/Unet3D_merge_tiny.py - 3D U-Net for satellite images"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, s, p, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, k, s, p, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(True)
        )
        self.residual = nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False)
    
    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, k, s):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(k, s),
            Conv3dBlock(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, x1_in, x2_in, out_ch, k, s, p):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(x1_in, x1_in, k, s, p, bias=True),
            nn.ReLU(True)
        )
        self.conv = Conv3dBlock(x1_in + x2_in, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffT = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,
            diffH // 2, diffH - diffH // 2,
            diffT // 2, diffT - diffT // 2
        ])
        
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch_list, out_ch):
        super().__init__()
        self.up_list = nn.ModuleList()
        
        for i, ch in enumerate(in_ch_list[:-1]):
            scale = int(np.power(2, len(in_ch_list) - 1 - i))
            self.up_list.append(nn.Sequential(
                nn.ConvTranspose3d(ch, ch, [1, scale, scale], [1, scale, scale]),
                nn.ReLU(True),
                nn.Conv3d(ch, in_ch_list[-1], 3, 1, 1),
                nn.BatchNorm3d(in_ch_list[-1]),
                nn.ReLU(True)
            ))
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_ch_list[-1], out_ch, 3, 1, 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d((1, 64, 64)),
            nn.Conv3d(out_ch, out_ch, 1, 1, 0)
        )
    
    def forward(self, x_list):
        x6, x7, x8, x9 = x_list
        x6 = self.up_list[0](x6)
        x7 = self.up_list[1](x7)
        x8 = self.up_list[2](x8)
        
        target_t = x9.size(2)
        x6 = F.interpolate(x6, size=(target_t, 64, 64))
        x7 = F.interpolate(x7, size=(target_t, 64, 64))
        x8 = F.interpolate(x8, size=(target_t, 64, 64))
        
        return self.final_conv(torch.cat([x6, x7, x8, x9], dim=2))


class Unet3D(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super().__init__()
        self.inc = Conv3dBlock(in_channel, 16)
        self.down1 = Down(16, 32, [1, 2, 2], [1, 2, 2])
        self.down2 = Down(32, 64, [1, 2, 2], [1, 2, 2])
        self.down3 = Down(64, 128, [2, 2, 2], [2, 2, 2])
        self.down4 = Down(128, 128, [2, 2, 2], [2, 2, 2])
        
        self.up1 = Up(128, 128, 64, [2, 2, 2], [2, 2, 2], 0)
        self.up2 = Up(64, 64, 32, [2, 2, 2], [2, 2, 2], 0)
        self.up3 = Up(32, 32, 16, [1, 2, 2], [1, 2, 2], 0)
        self.up4 = Up(16, 16, 16, [1, 2, 2], [1, 2, 2], 0)
        
        self.outc = OutConv([64, 32, 16, 16], out_channel)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        
        return self.outc([x6, x7, x8, x9])