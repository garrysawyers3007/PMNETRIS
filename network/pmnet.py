
from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

# Conv, Batchnorm, Relu layers, basic building block.
class _ConvBnReLU(nn.Sequential):

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

# Bottleneck layer cinstructed from ConvBnRelu layer block, buiding block for Res layers
class _Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

# Res Layer used to costruct the encoder
class _ResLayer(nn.Sequential):

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )

# Stem layer is the initial interfacing layer
class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch, in_ch = 2):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(in_ch, 2, 1, ceil_mode=True))



class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


# Atrous spatial pyramid pooling
class _ASPP(nn.Module):

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)



# Decoder layer constricted using these 2 blocks
def ConRu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

def ConRuT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

class FiLMModulation(nn.Module):
    def __init__(self, num_features, mlp_output_dim):
        super().__init__()
        self.gamma_layer = nn.Linear(mlp_output_dim, num_features)
        self.beta_layer = nn.Linear(mlp_output_dim, num_features)

    def forward(self, x, cond_vec):
        # x: [B, C, H, W]
        # cond_vec: [B, D]
        gamma = self.gamma_layer(cond_vec).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_layer(cond_vec).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return gamma * x + beta


class MLPConditioner(nn.Module):
    def __init__(self, in_features, hidden_dim=128, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.net(x)  # Output shape: [B, out_features]

class PMNetFiLM(nn.Module):

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=8):
        super(PMNetFiLM, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)

        # Decoder
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512+512, 512, 3, 1)
        self.conv_up3 = ConRuT(512+512, 256, 3, 1)
        self.conv_up2 = ConRu(256+256, 256, 3, 1)
        self.conv_up1 = ConRu(256+256, 256, 3, 1)

        self.conv_up0 = ConRu(256+64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
                         nn.Conv2d(128+2, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x, vec):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        cond_out = self.conditioner(vec)  # [B, 512]
        x8 = self.film(x8, cond_out)

        # Decoder
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        xup00 = self.conv_up00(xup0)

        return xup00
    

class PMNetFiLMNew(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=8):
        super(PMNetFiLMNew, self).__init__()

        # --- Stride Config ---
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)

        # --- Encoder Configuration ---
        # ch = [64, 128, 256, 512, 1024, 2048]
        ch = [64 * 2 ** p for p in range(6)]
        
        self.layer1 = _Stem(ch[0], in_ch=3)  # Output: 64 channels (x1)
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]) # Output: 256 channels (x2)
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1) # Output: 256 channels (x3)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]) # Output: 512 channels (x4)
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2]) # Output: 512 channels (x5)
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids) # Output: 1024 channels (x6)
        
        self.aspp = _ASPP(ch[4], 256, atrous_rates) # Output: 256 channels (x7)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.fc1 = _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1) # Output: 512 channels (x8)

        # --- Decoder Configuration (Fixed Channels) ---
        
        # Block 5: Inputs x8 (512)
        self.conv_up5 = ConRu(512, 512, 3, 1) 
        
        # Block 4: Inputs xup5 (512) + x5 (512) = 1024
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1) 
        
        # Block 3: Inputs xup4 (512) + x4 (512) = 1024
        self.conv_up3 = ConRu(512 + 512, 256, 3, 1) 
        
        # Block 2: Inputs xup3 (256) + x3 (256) = 512
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1) 
        
        # Block 1: Inputs xup2 (256) + x2 (256) = 512
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        
        # Block 0: Inputs xup1 (256) + x1 (64) = 320
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)

        # Final Block: Inputs xup0 (128) + Original Image (3) = 131
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(128 + 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, vec):
        # Encoder
        x1 = self.layer1(x)       # [B, 64, H/2, W/2]
        x2 = self.layer2(x1)      # [B, 256, H/4, W/4]
        x3 = self.reduce(x2)      # [B, 256, H/4, W/4]
        x4 = self.layer3(x3)      # [B, 512, H/8, W/8]
        x5 = self.layer4(x4)      # [B, 512, H/8, W/8] (if stride=1)
        x6 = self.layer5(x5)      # [B, 1024, H/8, W/8]
        x7 = self.aspp(x6)        # [B, 256, H/8, W/8]
        x8 = self.fc1(x7)         # [B, 512, H/8, W/8]

        # FiLM
        cond_out = self.conditioner(vec)
        x8 = self.film(x8, cond_out)

        # Decoder (with Robust Interpolation)
        xup5 = self.conv_up5(x8)
        if xup5.shape[2:] != x5.shape[2:]:
            xup5 = F.interpolate(xup5, size=x5.shape[2:], mode='bilinear', align_corners=False)
        xup5 = torch.cat([xup5, x5], dim=1) # 512+512 = 1024

        xup4 = self.conv_up4(xup5)
        if xup4.shape[2:] != x4.shape[2:]:
            xup4 = F.interpolate(xup4, size=x4.shape[2:], mode='bilinear', align_corners=False)
        xup4 = torch.cat([xup4, x4], dim=1) # 512+512 = 1024

        xup3 = self.conv_up3(xup4)
        if xup3.shape[2:] != x3.shape[2:]:
            xup3 = F.interpolate(xup3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        xup3 = torch.cat([xup3, x3], dim=1) # 256+256 = 512

        xup2 = self.conv_up2(xup3)
        if xup2.shape[2:] != x2.shape[2:]:
            xup2 = F.interpolate(xup2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        xup2 = torch.cat([xup2, x2], dim=1) # 256+256 = 512

        xup1 = self.conv_up1(xup2)
        if xup1.shape[2:] != x1.shape[2:]:
            xup1 = F.interpolate(xup1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        xup1 = torch.cat([xup1, x1], dim=1) # 256+64 = 320

        xup0 = self.conv_up0(xup1) # Output 128

        # Force 10x10 Output
        target_size = (10, 10)
        xup0 = F.interpolate(xup0, size=target_size, mode="bilinear", align_corners=False)
        x_resized = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        
        xup0 = torch.cat([xup0, x_resized], dim=1) # 128+3 = 131
        xup00 = self.conv_up00(xup0)

        return xup00
    

class PMNet(nn.Module):

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride):
        super(PMNet, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)

        # Decoder
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512+512, 512, 3, 1)
        self.conv_up3 = ConRuT(512+512, 256, 3, 1)
        self.conv_up2 = ConRu(256+256, 256, 3, 1)
        self.conv_up1 = ConRu(256+256, 256, 3, 1)

        self.conv_up0 = ConRu(256+64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
                         nn.Conv2d(128+2, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # Decoder
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        xup00 = self.conv_up00(xup0)
        
        return xup00
