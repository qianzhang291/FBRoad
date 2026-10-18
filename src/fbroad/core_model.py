"""One CMX FFM at the CMMPNet bottleneck; all original paths remain."""
import torch
from torch import nn
from ._vendor.cmmpnet.networks.CMMPNet import DinkNet34_CMMPNet
from .cmx_ffm import FeatureFusionModule

class CMMPNetCMX(DinkNet34_CMMPNet):
    def __init__(self, seed=23456, pretrained=False):
        super().__init__(pretrained=pretrained)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.cmx_fusion = FeatureFusionModule(dim=512, reduction=1, num_heads=8)
        self.alpha_rgb = nn.Parameter(torch.zeros(()))
        self.alpha_gps = nn.Parameter(torch.zeros(()))
        self.fusion_enabled = True

    def forward(self, inputs):

        x   = inputs[:,:3,:,:]    # image
        add = inputs[:,3:,:,:]    # gps_map or lidar_map

        x   = self.firstconv1(x)
        add = self.firstconv1_add(add)
        x   = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))

        x_e1   = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1, add_e1 = self.dem_e1(x_e1, add_e1)

        x_e2   = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        x_e2, add_e2 = self.dem_e2(x_e2, add_e2)

        x_e3   = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        x_e3, add_e3 = self.dem_e3(x_e3, add_e3)

        x_e4   = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)
        x_e4, add_e4 = self.dem_e4(x_e4, add_e4)

        # Center
        x_c   = self.dblock(x_e4)
        add_c = self.dblock_add(add_e4)
        if self.fusion_enabled:
            fused = self.cmx_fusion(x_c, add_c)
            x_c = x_c + self.alpha_rgb * fused
            add_c = add_c + self.alpha_gps * fused

        # Decoder
        x_d4   = self.decoder4(x_c) + x_e3
        add_d4 = self.decoder4_add(add_c) + add_e3
        x_d4, add_d4 = self.dem_d4(x_d4, add_d4)

        x_d3   = self.decoder3(x_d4) + x_e2
        add_d3 = self.decoder3_add(add_d4) + add_e2
        x_d3, add_d3 = self.dem_d3(x_d3, add_d3)

        x_d2   = self.decoder2(x_d3) + x_e1
        add_d2 = self.decoder2_add(add_d3) + add_e1
        x_d2, add_d2 = self.dem_d2(x_d2, add_d2)

        x_d1   = self.decoder1(x_d2)
        add_d1 = self.decoder1_add(add_d2)
        x_d1, add_d1 = self.dem_d1(x_d1, add_d1)

        x_out   = self.finalrelu1(self.finaldeconv1(x_d1))
        add_out = self.finalrelu1_add(self.finaldeconv1_add(add_d1))
        x_out   = self.finalrelu2(self.finalconv2(x_out))
        add_out = self.finalrelu2_add(self.finalconv2_add(add_out))

        out = self.finalconv(torch.cat((x_out, add_out),1)) # b*1*h*w
        return torch.sigmoid(out)
