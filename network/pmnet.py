
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
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=8, use_film=True):
        super(PMNetFiLMNew, self).__init__()
        self.use_film = use_film

        # --- Stride Config ---
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Only initialize FiLM components if use_film is True
        if self.use_film:
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

    def forward(self, x, vec=None):
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
        if self.use_film:
            if vec is None:
                # If FiLM is enabled but no vector is provided, you might want to raise an error
                # or just skip. Raising error ensures we don't silently fail.
                raise ValueError("Model initialized with use_film=True, but no conditioning vector 'vec' provided in forward().")
            
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


class PMNetFilMModified(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=7):
        super(PMNetFilMModified, self).__init__()

        # --- Stride Config ---
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

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
        
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)
        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)

        # --- Decoder Configuration (Fixed Channels) ---
        
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x, vec=None):
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

        # Decoder 
        h = x8
        h = F.adaptive_avg_pool2d(h, (10, 10))
        out = self.head(h)   # small conv stack: 512->128->1
        return out

class SpatialBottleneckModulation(nn.Module):
    """
    Spatially adaptive bottleneck modulation.

    Inputs:
      x   : [B, C, H, W]   bottleneck feature map
      vec : [B, D]         conditioning vector

    Process:
      1. vec -> conditioner -> cond_embed [B, E]
      2. broadcast cond_embed to [B, E, H, W]
      3. concatenate with x
      4. fuse using convs
      5. predict spatial gamma/beta: [B, C, H, W]
      6. apply residual modulation: gamma * x + beta
    """
    def __init__(self, num_features, cond_features, cond_embed_dim=128, hidden_channels=256):
        super().__init__()

        self.conditioner = MLPConditioner(
            in_features=cond_features,
            hidden_dim=128,
            out_features=cond_embed_dim
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(num_features + cond_embed_dim, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.gamma_layer = nn.Conv2d(hidden_channels, num_features, kernel_size=1)
        self.beta_layer = nn.Conv2d(hidden_channels, num_features, kernel_size=1)

    def forward(self, x, vec):
        b, c, h, w = x.shape

        cond = self.conditioner(vec)                    # [B, E]
        cond = cond.unsqueeze(-1).unsqueeze(-1)        # [B, E, 1, 1]
        cond = cond.expand(-1, -1, h, w)               # [B, E, H, W]

        fused = torch.cat([x, cond], dim=1)            # [B, C+E, H, W]
        fused = self.fuse(fused)                       # [B, hidden, H, W]

        gamma = self.gamma_layer(fused)                # [B, C, H, W]
        beta = self.beta_layer(fused)                  # [B, C, H, W]

        # Residual-style modulation
        out = gamma * x + beta
        return out


class PMNetSpatialFiLM(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=7):
        super(PMNetSpatialFiLM, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]


        ch = [64 * 2 ** p for p in range(6)]

        self.layer1 = _Stem(ch[0], in_ch=3)
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)

        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.fc1 = _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1)

        self.film = SpatialBottleneckModulation(
                num_features=512,
                cond_features=cond_features,
                cond_embed_dim=128,
                hidden_channels=256
            )

        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )


    def forward(self, x, vec=None):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        x8 = self.film(x8, vec)

        h = F.adaptive_avg_pool2d(x8, (10, 10))
        out = self.head(h)
        return out

class PMNetBaseline(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride):
        super(PMNetBaseline, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]


        ch = [64 * 2 ** p for p in range(6)]

        self.layer1 = _Stem(ch[0], in_ch=3)
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)

        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.fc1 = _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1)

        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )


    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        h = F.adaptive_avg_pool2d(x8, (10, 10))
        out = self.head(h)
        return out

class PMNet4Ch(nn.Module):
    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride):
        super(PMNet4Ch, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]

        self.layer1 = _Stem(ch[0], in_ch=4)
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids)

        self.aspp = _ASPP(ch[4], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.fc1 = _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1)

        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        h = F.adaptive_avg_pool2d(x8, (10, 10))
        out = self.head(h)
        return out


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


class PMNetFiLMCrop(nn.Module):
    """
    PMNet backbone with FiLM modulation at the bottleneck.
    Produces a 10x10 crop of the full-resolution output centered at the
    provided RX pixel coordinate.

    Args:
        n_blocks      : list of 4 ints  — ResLayer block counts
        atrous_rates  : list of ints    — ASPP dilation rates
        multi_grids   : list of ints    — multi-grid for layer5
        output_stride : int             — 8 or 16
        cond_features : int             — length of the FiLM conditioning vector

    Forward:
        x   : [B, 3, H, W]  — stacked input maps: [city_map, tx_map, rx_map]
        vec : [B, cond_features]  — conditioning vector

    Returns:
        [B, 1, 10, 10]  — cropped power map patch centred on the RX position
                          derived from channel 2 (rx_map) via centre-of-mass
    """

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=7,
                 backbone_checkpoint=None):
        super(PMNetFiLMCrop, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # FiLM components
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)
        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)

        # Encoder  (identical to PMNet, 3-channel input: city_map, tx_map, rx_map)
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

        # Decoder  (identical to PMNet)
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(128 + 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

        if backbone_checkpoint is not None:
            ckpt = torch.load(backbone_checkpoint, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            film_keys = {"conditioner", "film"}
            non_film_missing = [k for k in missing if k.split(".")[0] not in film_keys]
            if non_film_missing:
                print(f"PMNetFiLMCrop: backbone keys not found in checkpoint: {non_film_missing}")
            if unexpected:
                print(f"PMNetFiLMCrop: checkpoint keys not in model (ignored): {unexpected}")

    def forward(self, x, vec):
        # x: [B, 3, H, W] — channels: city_map, tx_map, rx_map
        # Only city_map and tx_map are fed to the encoder/decoder;
        # rx_map (channel 2) is used only to derive the crop centre.
        x_enc = x[:, :2]  # [B, 2, H, W]

        # Encoder
        x1 = self.layer1(x_enc)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # FiLM modulation at the bottleneck

        cond_out = self.conditioner(vec)   # [B, 64]
        x8 = self.film(x8, cond_out)      # [B, 512, h, w]

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

        xup0 = F.interpolate(xup0, size=x_enc.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x_enc], dim=1)
        full_output = self.conv_up00(xup0)  # [B, 1, H, W]

        rx_pixel = self._extract_rx_center(x[:, 2])  # derive from rx_map channel
        return self._crop_at_rx(full_output, rx_pixel, crop_size=14)

    @staticmethod
    def _extract_rx_center(rx_map):
        """
        Computes the centre-of-mass of the RX marker in rx_map.

        Args:
            rx_map : [B, H, W]  — single-channel RX map (white square on black)

        Returns:
            [B, 2] long tensor of (row, col) pixel coordinates
        """
        B, H, W = rx_map.shape
        # Use float mask; the marker is non-zero where the RX box is drawn
        mask = rx_map.float()  # [B, H, W]
        total = mask.sum(dim=(1, 2)).clamp(min=1.0)  # [B]

        row_idx = torch.arange(H, device=rx_map.device).float()  # [H]
        col_idx = torch.arange(W, device=rx_map.device).float()  # [W]

        row_center = (mask * row_idx.view(1, H, 1)).sum(dim=(1, 2)) / total  # [B]
        col_center = (mask * col_idx.view(1, 1, W)).sum(dim=(1, 2)) / total  # [B]

        return torch.stack([row_center, col_center], dim=1).round().long()  # [B, 2]

    @staticmethod
    def _crop_at_rx(feature_map, rx_pixel, crop_size=14):
        """
        Crops a (crop_size x crop_size) patch from feature_map centred at each
        per-sample RX pixel coordinate.

        Boundary handling: zero-pad the map by half=crop_size//2 on all sides so
        that the crop is always exactly centred on rx_pixel, regardless of how
        close it is to the border.

        Args:
            feature_map : [B, C, H, W]
            rx_pixel    : [B, 2]  integer (row, col) in the original map
            crop_size   : int (must be even)

        Returns:
            [B, C, crop_size, crop_size]
        """
        half = crop_size // 2
        # Pad: left, right, top, bottom  (F.pad order is last-dim first)
        padded = F.pad(feature_map, [half, half, half, half])  # [B, C, H+cs, W+cs]

        crops = []
        for b in range(feature_map.shape[0]):
            r = int(rx_pixel[b, 0].item()) + half   # position of RX in padded image
            c = int(rx_pixel[b, 1].item()) + half
            crops.append(padded[b:b+1, :, r - half : r + half, c - half : c + half])

        return torch.cat(crops, dim=0)  # [B, C, crop_size, crop_size]


class PMNetFiLMSoftCrop(nn.Module):
    """
    PMNet backbone with FiLM modulation at the bottleneck.
    Crops a 32x32 patch centred at the RX position (derived from rx_map via
    centre-of-mass) then refines it to a 10x10 output through a learned roi_head.

    Args:
        n_blocks      : list of 4 ints  — ResLayer block counts
        atrous_rates  : list of ints    — ASPP dilation rates
        multi_grids   : list of ints    — multi-grid for layer5
        output_stride : int             — 8 or 16
        cond_features : int             — length of the FiLM conditioning vector
        backbone_checkpoint : str|None  — path to a pre-trained PMNet checkpoint

    Forward:
        x   : [B, 3, H, W]  — stacked input maps: [city_map, tx_map, rx_map]
        vec : [B, cond_features]  — conditioning vector

    Returns:
        [B, 1, 10, 10]  — refined power map patch centred on the RX position
    """

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=7,
                 backbone_checkpoint=None):
        super(PMNetFiLMSoftCrop, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # FiLM components
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)
        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)

        # Encoder  (2-channel input: city_map, tx_map)
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

        # Decoder  (identical to PMNet)
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)
        # RoI refinement head: 32x32 crop from xup0 (128ch) -> 10x10 output
        self.roi_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(64, 1, 1),
        )

        if backbone_checkpoint is not None:
            ckpt = torch.load(backbone_checkpoint, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            new_keys = {"conditioner", "film", "roi_head"}
            removed_keys = {"conv_up00"}
            non_new_missing = [k for k in missing if k.split(".")[0] not in new_keys]
            unexpected_real = [k for k in unexpected if k.split(".")[0] not in removed_keys]
            if non_new_missing:
                print(f"PMNetFiLMSoftCrop: backbone keys not found in checkpoint: {non_new_missing}")
            if unexpected_real:
                print(f"PMNetFiLMSoftCrop: checkpoint keys not in model (ignored): {unexpected_real}")

    def forward(self, x, vec):
        # x: [B, 3, H, W] — channels: city_map, tx_map, rx_map
        x_enc = x[:, :2]  # [B, 2, H, W]

        # Encoder
        x1 = self.layer1(x_enc)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # FiLM modulation at the bottleneck
        cond_out = self.conditioner(vec)   # [B, 64]
        x8 = self.film(x8, cond_out)      # [B, 512, h, w]

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

        xup0 = F.interpolate(xup0, size=x_enc.shape[2:], mode="bilinear", align_corners=False)

        rx_pixel = PMNetFiLMCrop._extract_rx_center(x[:, 2])
        crop32 = PMNetFiLMCrop._crop_at_rx(xup0, rx_pixel, crop_size=32)  # [B, 128, 32, 32]
        return self.roi_head(crop32)  # [B, 1, 10, 10]

class PMNetFiLMSoftCropV2(nn.Module):
    """
    PMNet backbone with FiLM modulation at the bottleneck.
    Crops a 32x32 patch centred at the RX position (derived from rx_map via
    centre-of-mass) then refines it to a 10x10 output through a learned roi_head.

    Args:
        n_blocks      : list of 4 ints  — ResLayer block counts
        atrous_rates  : list of ints    — ASPP dilation rates
        multi_grids   : list of ints    — multi-grid for layer5
        output_stride : int             — 8 or 16
        cond_features : int             — length of the FiLM conditioning vector
        backbone_checkpoint : str|None  — path to a pre-trained PMNet checkpoint

    Forward:
        x   : [B, 3, H, W]  — stacked input maps: [city_map, tx_map, rx_map]
        vec : [B, cond_features]  — conditioning vector

    Returns:
        [B, 1, 10, 10]  — refined power map patch centred on the RX position
    """

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=7, sigma=32.0, crop_size=32,
                 backbone_checkpoint=None):
        super(PMNetFiLMSoftCropV2, self).__init__()
        self.crop_size = crop_size

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # FiLM components
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)
        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)

        # Encoder  (2-channel input: city_map, tx_map)
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

        # Decoder  (identical to PMNet)
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)
        # # RoI refinement head: 32x32 crop from xup0 (128ch) -> 10x10 output

        self.roi_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(64, 1, 1),
        )

        self.sigma = sigma
        if backbone_checkpoint is not None:
            ckpt = torch.load(backbone_checkpoint, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            new_keys = {"conditioner", "film", "roi_head"}
            removed_keys = {"conv_up00"}
            non_new_missing = [k for k in missing if k.split(".")[0] not in new_keys]
            unexpected_real = [k for k in unexpected if k.split(".")[0] not in removed_keys]
            if non_new_missing:
                print(f"PMNetFiLMSoftCrop: backbone keys not found in checkpoint: {non_new_missing}")
            if unexpected_real:
                print(f"PMNetFiLMSoftCrop: checkpoint keys not in model (ignored): {unexpected_real}")

    def forward(self, x, vec):
        # x: [B, 3, H, W] — channels: city_map, tx_map, rx_map
        x_enc = x[:, :2]  # [B, 2, H, W]

        # Encoder
        x1 = self.layer1(x_enc)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # FiLM modulation at the bottleneck
        cond_out = self.conditioner(vec)   # [B, 64]
        x8 = self.film(x8, cond_out)      # [B, 512, h, w]

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

        xup0 = F.interpolate(xup0, size=x_enc.shape[2:], mode="bilinear", align_corners=False)

        rx_pixel = self._extract_rx_center_differentiable(x[:, 2])
        gate = self.make_rx_gaussian(rx_pixel, xup0.shape[-2], xup0.shape[-1], sigma=self.sigma)
        xup0_gated = xup0 * gate
        crop_gated = self._crop_at_rx_differentiable(xup0_gated, rx_pixel, crop_size=self.crop_size)
        return self.roi_head(crop_gated)  # [B, 1, 10, 10]
    
    def make_rx_gaussian(self, rx_pixel, H, W, sigma):
        """
        Create a Gaussian attention map centered at RX.

        Args:
            rx_pixel : [B, 2] tensor with (y, x) coordinates
            H, W     : spatial size
            sigma    : std deviation (controls spread)

        Returns:
            gate     : [B, 1, H, W]
        """
        device = rx_pixel.device
        B = rx_pixel.shape[0]

        # Create coordinate grid
        y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

        # RX center
        cy = rx_pixel[:, 0].view(B, 1, 1)
        cx = rx_pixel[:, 1].view(B, 1, 1)

        # Squared distance
        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2

        # Gaussian
        gate = torch.exp(-dist_sq / (2 * sigma ** 2))

        return gate.unsqueeze(1)  # [B, 1, H, W]

    def _extract_rx_center_differentiable(self, rx_map, beta=50.0):
        """
        Differentiable centre-of-mass via spatial softmax.
        Returns float coordinates that carry gradients (no .round().long()).

        Args:
            rx_map : [B, H, W] or [B, 1, H, W]
            beta   : softmax temperature — higher → closer to argmax

        Returns:
            [B, 2] float tensor of (row, col) coordinates
        """
        if rx_map.dim() == 3:
            rx_map = rx_map.unsqueeze(1)
        B, _, H, W = rx_map.shape
        device = rx_map.device

        weights = F.softmax(rx_map.view(B, -1) * beta, dim=-1).view(B, H, W)

        grid_y = torch.arange(H, device=device).float().view(1, H, 1)
        grid_x = torch.arange(W, device=device).float().view(1, 1, W)

        cy = (weights * grid_y).sum(dim=(1, 2))  # [B]
        cx = (weights * grid_x).sum(dim=(1, 2))  # [B]

        return torch.stack([cy, cx], dim=1)  # [B, 2]

    def _crop_at_rx_differentiable(self, feature_map, rx_pixel, crop_size=32):
        """
        Differentiable crop centred at rx_pixel using F.grid_sample.

        Args:
            feature_map : [B, C, H, W]
            rx_pixel    : [B, 2] floats (row, col) — carries gradients
            crop_size   : int — output spatial size

        Returns:
            [B, C, crop_size, crop_size]
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # Normalise RX centre to [-1, 1] (grid_sample convention)
        norm_y = (rx_pixel[:, 0] / (H - 1)) * 2 - 1  # [B]
        norm_x = (rx_pixel[:, 1] / (W - 1)) * 2 - 1  # [B]

        # Local grid spanning crop_size pixels, scaled to fraction of full map
        side = torch.linspace(-1, 1, crop_size, device=device)
        mesh_y, mesh_x = torch.meshgrid(side * (crop_size / H), side * (crop_size / W), indexing='ij')
        grid_base = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0)  # [1, cs, cs, 2]

        # Shift to RX position — addition propagates grads through norm_x/norm_y
        offset = torch.stack([norm_x, norm_y], dim=-1).view(B, 1, 1, 2)
        grid = grid_base + offset  # [B, cs, cs, 2]

        return F.grid_sample(feature_map, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

class PMNetFiLMSoftCropV3(nn.Module):
    """
    PMNet backbone with FiLM modulation at the bottleneck.
    Applies a Gaussian attention gate centred at the RX position (derived from
    rx_map via centre-of-mass) to the full-resolution decoder feature map
    (xup0, 128ch), then downsamples to a 10x10 output via a strided-conv
    roi_head. No explicit crop is performed; the Gaussian gate suppresses
    activations far from the RX location.

    Args:
        n_blocks      : list of 4 ints  — ResLayer block counts
        atrous_rates  : list of ints    — ASPP dilation rates
        multi_grids   : list of ints    — multi-grid for layer5
        output_stride : int             — 8 or 16
        cond_features : int             — length of the FiLM conditioning vector
        sigma         : float           — std deviation of the RX Gaussian gate
        crop_size     : int             — unused; retained for API compatibility
        backbone_checkpoint : str|None  — path to a pre-trained PMNet checkpoint

    Forward:
        x   : [B, 3, H, W]  — stacked input maps: [city_map, tx_map, rx_map]
        vec : [B, cond_features]  — conditioning vector

    Returns:
        [B, 1, 10, 10]  — power map prediction centred on the RX position
    """

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride, cond_features=7, sigma=64.0, crop_size=32,
                 backbone_checkpoint=None):
        super(PMNetFiLMSoftCropV3, self).__init__()
        self.crop_size = crop_size

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # FiLM components
        self.conditioner = MLPConditioner(in_features=cond_features, out_features=64)
        self.film = FiLMModulation(num_features=512, mlp_output_dim=64)

        # Encoder  (2-channel input: city_map, tx_map)
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

        # Decoder  (identical to PMNet)
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)
        # RoI refinement head: 32x32 crop from xup0 (128ch) -> 10x10 output
        self.roi_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),   # 256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 96, 3, stride=2, padding=1),    # 128 -> 64
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 64, 3, stride=2, padding=1),     # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),     # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(64, 1, 1)
        )

        self.sigma = sigma
        if backbone_checkpoint is not None:
            ckpt = torch.load(backbone_checkpoint, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            new_keys = {"conditioner", "film", "roi_head"}
            removed_keys = {"conv_up00"}
            non_new_missing = [k for k in missing if k.split(".")[0] not in new_keys]
            unexpected_real = [k for k in unexpected if k.split(".")[0] not in removed_keys]
            if non_new_missing:
                print(f"PMNetFiLMSoftCrop: backbone keys not found in checkpoint: {non_new_missing}")
            if unexpected_real:
                print(f"PMNetFiLMSoftCrop: checkpoint keys not in model (ignored): {unexpected_real}")

    def forward(self, x, vec):
        # x: [B, 3, H, W] — channels: city_map, tx_map, rx_map
        x_enc = x[:, :2]  # [B, 2, H, W]

        # Encoder
        x1 = self.layer1(x_enc)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # FiLM modulation at the bottleneck
        cond_out = self.conditioner(vec)   # [B, 64]
        x8 = self.film(x8, cond_out)      # [B, 512, h, w]

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

        xup0 = F.interpolate(xup0, size=x_enc.shape[2:], mode="bilinear", align_corners=False)

        rx_pixel = PMNetFiLMCrop._extract_rx_center(x[:, 2])
        gate = self.make_rx_gaussian(rx_pixel, xup0.shape[-2], xup0.shape[-1], sigma=self.sigma)
        xup0_gated = xup0 * gate

        return self.roi_head(xup0_gated) # [B, 1, 10, 10]
    
    def make_rx_gaussian(self, rx_pixel, H, W, sigma):
        """
        Create a Gaussian attention map centered at RX.

        Args:
            rx_pixel : [B, 2] tensor with (y, x) coordinates
            H, W     : spatial size
            sigma    : std deviation (controls spread)

        Returns:
            gate     : [B, 1, H, W]
        """
        device = rx_pixel.device
        B = rx_pixel.shape[0]

        # Create coordinate grid
        y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

        # RX center
        cy = rx_pixel[:, 0].view(B, 1, 1)
        cx = rx_pixel[:, 1].view(B, 1, 1)

        # Squared distance
        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2

        # Gaussian
        gate = torch.exp(-dist_sq / (2 * sigma ** 2))

        return gate.unsqueeze(1)  # [B, 1, H, W]

class GeometryEncoder(nn.Module):
    """Small CNN that encodes 3-channel spatial geometry input
    [tx_map, ris_map, rx_map] down to a feature map matching the
    encoder bottleneck spatial resolution.

    256×256 → 128×128 (stride 2) → 64×64 (stride 2) → 64×64 (stride 1)
    """

    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 24, kernel_size=3, stride=2, padding=1),   # 256 -> 128
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),      # 128 -> 64
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, out_ch, kernel_size=3, stride=1, padding=2, dilation=2),  # 64 -> 64
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.net(x)


class PMNetFiLMSoftCropV4_1(nn.Module):
    """PMNet with spatial geometry conditioning at the bottleneck.

    Replaces the MLP+FiLM vector conditioning of V2 with a GeometryEncoder
    CNN branch that processes spatial maps [tx_map, ris_map, rx_map] and
    fuses with the encoder bottleneck via a zero-initialised residual
    weighted-sum.  The pretrained encoder and decoder are unchanged.

    Forward:
        x       : [B, 3, H, W]  — [city_map, tx_map, rx_map]
        ris_map : [B, 1, H, W]  — RIS spatial map (zeros for noRIS)

    Returns:
        [B, 1, 10, 10]  — power map patch centred on the RX position
    """

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride,
                 sigma=32.0, crop_size=32, backbone_checkpoint=None):
        super(PMNetFiLMSoftCropV4_1, self).__init__()
        self.crop_size = crop_size
        self.sigma = sigma

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # —— Geometry conditioning branch (NEW) ——
        # self.geom_encoder = GeometryEncoder(in_ch=3, out_ch=128)
        self.geom_encoder = GeometryEncoder(in_ch=3, out_ch=96)
        # self.geom_proj = nn.Sequential(
        #     nn.Conv2d(128, 512, kernel_size=1),
        #     nn.BatchNorm2d(512),
        # )

        self.geom_proj = nn.Sequential(
            nn.Conv2d(96, 512, kernel_size=1),
            nn.BatchNorm2d(512),
        )
        # Fuse encoder bottleneck (512) + geometry projection (512) → 512
        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # —— Encoder (2-channel: city_map + tx_map, unchanged from V2) ——
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

        # —— Decoder (unchanged from V2) ——
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)

        # RoI refinement head: crop from xup0 (128ch) → 10×10 output
        # self.roi_head = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((10, 10)),
        #     nn.Conv2d(64, 1, 1),
        # )

        self.roi_head = nn.Sequential(
            nn.Conv2d(256, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(48, 1, 1),
        )

        # —— Optional pretrained backbone loading ——
        if backbone_checkpoint is not None:
            ckpt = torch.load(backbone_checkpoint, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            new_keys = {"geom_encoder", "geom_proj", "fuse_1x1", "roi_head"}
            removed_keys = {"conv_up00", "conditioner", "film"}
            non_new_missing = [k for k in missing if k.split(".")[0] not in new_keys]
            unexpected_real = [k for k in unexpected if k.split(".")[0] not in removed_keys]
            if non_new_missing:
                print(f"PMNetFiLMSoftCropV4: backbone keys not found in checkpoint: {non_new_missing}")
            if unexpected_real:
                print(f"PMNetFiLMSoftCropV4: checkpoint keys not in model (ignored): {unexpected_real}")

    def forward(self, x, ris_map):
        # x: [B, 3, H, W] — [city_map, tx_map, rx_map]
        # ris_map: [B, 1, H, W] — RIS spatial map (zeros for noRIS)
        x_enc = x[:, :2]  # [B, 2, H, W]

        # —— Encoder ——
        x1 = self.layer1(x_enc)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)  # [B, 512, h, w]

        # —— Geometry conditioning (replaces FiLM) ——
        geom_input = torch.cat([x[:, 1:2], ris_map, x[:, 2:3]], dim=1)  # [B, 3, H, W]
        geom_feat = self.geom_encoder(geom_input)  # [B, 128, h', w']
        # geom_feat = F.interpolate(geom_feat, size=x8.shape[2:],
        #                           mode='bilinear', align_corners=False)
        geom_feat = F.adaptive_max_pool2d(geom_feat, x8.shape[2:])
        geom_proj = self.geom_proj(geom_feat)  # [B, 512, h, w]
        fused = torch.cat([x8, geom_proj], dim=1)  # [B, 1024, h, w]
        x8 = self.fuse_1x1(fused)  # [B, 512, h, w]

        # —— Decoder ——
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

        xup0 = F.interpolate(xup0, size=x_enc.shape[2:],
                             mode="bilinear", align_corners=False)

        # —— RX-centred crop + RoI head ——
        rx_pixel = self._extract_rx_center_differentiable(x[:, 2])
        gate = self._make_rx_gaussian(rx_pixel, xup0.shape[-2],
                                      xup0.shape[-1], sigma=self.sigma)
        xup0_gated = xup0 * gate

        crop_gated = self._crop_at_rx_differentiable(
            xup0_gated, rx_pixel, crop_size=self.crop_size)
        crop_raw = self._crop_at_rx_differentiable(
            xup0, rx_pixel, crop_size=self.crop_size)

        roi_feat = torch.cat([crop_raw, crop_gated], dim=1)
        return self.roi_head(roi_feat)  # [B, 1, 10, 10]

    # —— Helpers (unchanged from V2) ——

    @staticmethod
    def _make_rx_gaussian(rx_pixel, H, W, sigma):
        device = rx_pixel.device
        B = rx_pixel.shape[0]
        y_coords = torch.arange(H, device=device).float().view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=device).float().view(1, 1, W).expand(B, H, W)
        cy = rx_pixel[:, 0].view(B, 1, 1)
        cx = rx_pixel[:, 1].view(B, 1, 1)
        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        gate = torch.exp(-dist_sq / (2 * sigma ** 2))
        return gate.unsqueeze(1)  # [B, 1, H, W]

    @staticmethod
    def _extract_rx_center_differentiable(rx_map, beta=50.0):
        if rx_map.dim() == 3:
            rx_map = rx_map.unsqueeze(1)
        B, _, H, W = rx_map.shape
        device = rx_map.device
        weights = F.softmax(rx_map.view(B, -1) * beta, dim=-1).view(B, H, W)
        grid_y = torch.arange(H, device=device).float().view(1, H, 1)
        grid_x = torch.arange(W, device=device).float().view(1, 1, W)
        cy = (weights * grid_y).sum(dim=(1, 2))
        cx = (weights * grid_x).sum(dim=(1, 2))
        return torch.stack([cy, cx], dim=1)  # [B, 2]

    @staticmethod
    def _crop_at_rx_differentiable(feature_map, rx_pixel, crop_size=32):
        B, C, H, W = feature_map.shape
        device = feature_map.device
        norm_y = (rx_pixel[:, 0] / (H - 1)) * 2 - 1
        norm_x = (rx_pixel[:, 1] / (W - 1)) * 2 - 1
        side = torch.linspace(-1, 1, crop_size, device=device)
        mesh_y, mesh_x = torch.meshgrid(
            side * (crop_size / H), side * (crop_size / W), indexing='ij')
        grid_base = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0)
        offset = torch.stack([norm_x, norm_y], dim=-1).view(B, 1, 1, 2)
        grid = grid_base + offset
        return F.grid_sample(feature_map, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)


class PMNetSoftCropBaseline(nn.Module):
    """Baseline PMNet with RX-centred soft crop — no geometry conditioning.

    Same encoder, decoder, Gaussian gating, differentiable crop, and roi_head
    as PMNetFiLMSoftCropV4_1, but without the GeometryEncoder/fusion branch.

    Forward:
        x : [B, 3, H, W]  — [city_map, tx_map, rx_map]

    Returns:
        [B, 1, 10, 10]  — power map patch centred on the RX position
    """

    def __init__(self, n_blocks, atrous_rates, multi_grids, output_stride,
                 sigma=32.0, crop_size=32, backbone_checkpoint=None):
        super(PMNetSoftCropBaseline, self).__init__()
        self.crop_size = crop_size
        self.sigma = sigma

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # —— Encoder (2-channel: city_map + tx_map) ——
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

        # —— Decoder ——
        self.conv_up5 = ConRu(512, 512, 3, 1)
        self.conv_up4 = ConRu(512 + 512, 512, 3, 1)
        self.conv_up3 = ConRuT(512 + 512, 256, 3, 1)
        self.conv_up2 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up1 = ConRu(256 + 256, 256, 3, 1)
        self.conv_up0 = ConRu(256 + 64, 128, 3, 1)

        # RoI refinement head
        self.roi_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Conv2d(64, 1, 1),
        )

        # —— Optional pretrained backbone loading ——
        if backbone_checkpoint is not None:
            ckpt = torch.load(backbone_checkpoint, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            new_keys = {"roi_head"}
            removed_keys = {"conv_up00", "conditioner", "film"}
            non_new_missing = [k for k in missing if k.split(".")[0] not in new_keys]
            unexpected_real = [k for k in unexpected if k.split(".")[0] not in removed_keys]
            if non_new_missing:
                print(f"PMNetSoftCropBaseline: backbone keys not found: {non_new_missing}")
            if unexpected_real:
                print(f"PMNetSoftCropBaseline: unexpected keys (ignored): {unexpected_real}")

    def forward(self, x):
        x_enc = x[:, :2]  # [B, 2, H, W]

        # —— Encoder ——
        x1 = self.layer1(x_enc)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)  # [B, 512, h, w]

        # —— Decoder ——
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

        xup0 = F.interpolate(xup0, size=x_enc.shape[2:],
                             mode="bilinear", align_corners=False)

        # —— RX-centred crop + RoI head ——
        rx_pixel = self._extract_rx_center_differentiable(x[:, 2])
        gate = self._make_rx_gaussian(rx_pixel, xup0.shape[-2],
                                      xup0.shape[-1], sigma=self.sigma)
        xup0_gated = xup0 * gate

        crop_gated = self._crop_at_rx_differentiable(
            xup0_gated, rx_pixel, crop_size=self.crop_size)
        crop_raw = self._crop_at_rx_differentiable(
            xup0, rx_pixel, crop_size=self.crop_size)

        roi_feat = torch.cat([crop_raw, crop_gated], dim=1)
        return self.roi_head(roi_feat)  # [B, 1, 10, 10]

    # —— Helpers ——

    @staticmethod
    def _make_rx_gaussian(rx_pixel, H, W, sigma):
        device = rx_pixel.device
        B = rx_pixel.shape[0]
        y_coords = torch.arange(H, device=device).float().view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=device).float().view(1, 1, W).expand(B, H, W)
        cy = rx_pixel[:, 0].view(B, 1, 1)
        cx = rx_pixel[:, 1].view(B, 1, 1)
        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        gate = torch.exp(-dist_sq / (2 * sigma ** 2))
        return gate.unsqueeze(1)

    @staticmethod
    def _extract_rx_center_differentiable(rx_map, beta=50.0):
        if rx_map.dim() == 3:
            rx_map = rx_map.unsqueeze(1)
        B, _, H, W = rx_map.shape
        device = rx_map.device
        weights = F.softmax(rx_map.view(B, -1) * beta, dim=-1).view(B, H, W)
        grid_y = torch.arange(H, device=device).float().view(1, H, 1)
        grid_x = torch.arange(W, device=device).float().view(1, 1, W)
        cy = (weights * grid_y).sum(dim=(1, 2))
        cx = (weights * grid_x).sum(dim=(1, 2))
        return torch.stack([cy, cx], dim=1)

    @staticmethod
    def _crop_at_rx_differentiable(feature_map, rx_pixel, crop_size=32):
        B, C, H, W = feature_map.shape
        device = feature_map.device
        norm_y = (rx_pixel[:, 0] / (H - 1)) * 2 - 1
        norm_x = (rx_pixel[:, 1] / (W - 1)) * 2 - 1
        side = torch.linspace(-1, 1, crop_size, device=device)
        mesh_y, mesh_x = torch.meshgrid(
            side * (crop_size / H), side * (crop_size / W), indexing='ij')
        grid_base = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0)
        offset = torch.stack([norm_x, norm_y], dim=-1).view(B, 1, 1, 2)
        grid = grid_base + offset
        return F.grid_sample(feature_map, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)
