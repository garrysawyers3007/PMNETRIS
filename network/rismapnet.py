import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Building blocks
# -----------------------------

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def norm_layer(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    # GroupNorm is usually more stable than BatchNorm for small datasets / small batch sizes.
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class FiLM(nn.Module):
    """
    Feature-wise linear modulation:
      y = (1 + gamma) * x + beta
    We use (1 + gamma) so zero init behaves like identity.
    """
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # gamma, beta shape: [B, C]
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * x + beta


class FiLMGenerator(nn.Module):
    """
    Generates per-stage (gamma, beta) from conditioning vector.
    """
    def __init__(self, cond_dim: int, stage_channels: List[int], hidden: int = 128):
        super().__init__()
        self.stage_channels = stage_channels

        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            SiLU(),
            nn.Linear(hidden, hidden),
            SiLU(),
        )
        self.to_gb = nn.ModuleList([
            nn.Linear(hidden, 2 * ch) for ch in stage_channels
        ])

        # Initialize to near-identity modulation
        for layer in self.to_gb:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, cond: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        h = self.trunk(cond)
        out = []
        for proj, ch in zip(self.to_gb, self.stage_channels):
            gb = proj(h)  # [B, 2C]
            gamma, beta = gb[:, :ch], gb[:, ch:]
            out.append((gamma, beta))
        return out


class ResBlock(nn.Module):
    """
    Conv -> GN -> SiLU -> Conv -> GN, with residual.
    Optional FiLM after each GN (commonly after the first GN is enough).
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_film: bool = False,
        num_groups: int = 8,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.use_film = use_film
        self.film = FiLM() if use_film else None

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = norm_layer(out_ch, num_groups)
        self.act1 = SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = norm_layer(out_ch, num_groups)
        self.act2 = SiLU()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, film_gb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        if self.use_film and film_gb is not None:
            gamma, beta = film_gb
            h = self.film(h, gamma, beta)
        h = self.act1(h)

        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        # (Optional) FiLM again here; often not necessary. Kept simple.
        h = self.act2(h)

        return h + residual


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # stride-2 conv downsample
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class CoordConcat(nn.Module):
    """
    Appends normalized coordinate channels (x,y) to the input.
    Helpful when location matters (propagation maps often benefit).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        yy = torch.linspace(-1, 1, steps=h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, steps=w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)


# -----------------------------
# Model
# -----------------------------

@dataclass
class RISMapNetConfig:
    in_channels: int = 3              # building, TX, RX
    out_channels: int = 1             # single-channel 10x10
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8, 8)  # stages -> 256->128->64->32->16
    blocks_per_stage: int = 1
    film_stages: Tuple[int, ...] = (2, 3, 4)          # apply FiLM at 64x64, 32x32, 16x16 (0-based stage idx)
    film_hidden: int = 128
    dropout: float = 0.05
    use_coordconv: bool = True
    use_dilated_block_at_16: bool = True
    target_hw: Tuple[int, int] = (10, 10)            # output grid
    num_groups: int = 8


class RISMapNet(nn.Module):
    """
    Encoder-only, FiLM-conditioned, outputs 10x10 map.
    Input:  B x 3 x 256 x 256
    Cond:   B x cond_dim  (e.g., [x, y, sin(theta), cos(theta)])
    Output: B x 1 x 10 x 10
    """
    def __init__(self, cond_dim: int, cfg: RISMapNetConfig):
        super().__init__()
        self.cfg = cfg
        self.coord = CoordConcat() if cfg.use_coordconv else nn.Identity()
        in_ch = cfg.in_channels + (2 if cfg.use_coordconv else 0)

        # Build stage channels
        stage_ch = [cfg.base_channels * m for m in cfg.channel_mults]  # len=5 for 256..16
        self.stage_ch = stage_ch

        self.stem = nn.Conv2d(in_ch, stage_ch[0], kernel_size=3, padding=1)

        # Encoder stages: at each stage, N resblocks then downsample (except last)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for s in range(len(stage_ch)):
            blocks = []
            in_c = stage_ch[s - 1] if s > 0 else stage_ch[0]
            out_c = stage_ch[s]
            # first block may change channels (except stage0)
            blocks.append(ResBlock(in_c, out_c, use_film=(s in cfg.film_stages),
                                   num_groups=cfg.num_groups, dropout=cfg.dropout))
            for _ in range(cfg.blocks_per_stage - 1):
                blocks.append(ResBlock(out_c, out_c, use_film=(s in cfg.film_stages),
                                       num_groups=cfg.num_groups, dropout=cfg.dropout))
            self.stages.append(nn.ModuleList(blocks))
            if s < len(stage_ch) - 1:
                self.downs.append(Downsample(out_c))

        # Optional dilated block at the 16x16 stage to enlarge receptive field
        self.dilated = None
        if cfg.use_dilated_block_at_16:
            c16 = stage_ch[-1]
            self.dilated = ResBlock(c16, c16, use_film=(len(stage_ch)-1 in cfg.film_stages),
                                    num_groups=cfg.num_groups, dropout=cfg.dropout, dilation=2)

        # FiLM generator only for stages you modulate
        film_channels = [stage_ch[s] for s in cfg.film_stages]
        self.film_gen = FiLMGenerator(cond_dim=cond_dim, stage_channels=film_channels, hidden=cfg.film_hidden)

        # Grid reducer and head
        self.pool = nn.AdaptiveAvgPool2d(cfg.target_hw)
        head_in = stage_ch[-1]
        self.head = nn.Sequential(
            nn.Conv2d(head_in, max(head_in // 2, 64), kernel_size=3, padding=1),
            SiLU(),
            nn.Conv2d(max(head_in // 2, 64), cfg.out_channels, kernel_size=1),
        )

        # Store mapping from stage idx -> index in film list
        self._film_stage_to_idx = {s: i for i, s in enumerate(cfg.film_stages)}
        self._film = FiLM()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, 3, 256, 256]
        cond: [B, cond_dim]  (recommend: [x_norm, y_norm, sinθ, cosθ])
        """
        x = self.coord(x)
        h = self.stem(x)

        film_gbs = self.film_gen(cond)  # list of (gamma,beta) aligned with cfg.film_stages

        for s, blocks in enumerate(self.stages):
            gb = None
            if s in self._film_stage_to_idx:
                gb = film_gbs[self._film_stage_to_idx[s]]
            for blk in blocks:
                h = blk(h, film_gb=gb)
            if s < len(self.downs):
                h = self.downs[s](h)

        if self.dilated is not None:
            s_last = len(self.stage_ch) - 1
            gb = None
            if s_last in self._film_stage_to_idx:
                gb = film_gbs[self._film_stage_to_idx[s_last]]
            h = self.dilated(h, film_gb=gb)

        h = self.pool(h)     # -> [B, D, 10, 10]
        y = self.head(h)     # -> [B, 1, 10, 10]
        return y


# -----------------------------
# Presets for ~6k samples
# -----------------------------

def make_model_preset(cond_dim: int, preset: str = "small") -> RISMapNet:
    """
    For ~6000 samples, start with 'small' or 'medium' unless your scenes are very diverse.
    """
    if preset == "small":
        cfg = RISMapNetConfig(
            base_channels=24,
            channel_mults=(1, 2, 4, 6, 6),  # 24,48,96,144,144
            blocks_per_stage=1,
            film_hidden=96,
            dropout=0.05,
            use_coordconv=True,
            use_dilated_block_at_16=True,
        )
    elif preset == "medium":
        cfg = RISMapNetConfig(
            base_channels=32,
            channel_mults=(1, 2, 4, 8, 8),  # 32,64,128,256,256
            blocks_per_stage=1,
            film_hidden=128,
            dropout=0.05,
            use_coordconv=True,
            use_dilated_block_at_16=True,
        )
    elif preset == "large":
        cfg = RISMapNetConfig(
            base_channels=40,
            channel_mults=(1, 2, 4, 8, 12),  # 40,80,160,320,480
            blocks_per_stage=2,
            film_hidden=192,
            dropout=0.10,
            use_coordconv=True,
            use_dilated_block_at_16=True,
        )
    else:
        raise ValueError("preset must be one of: small | medium | large")

    return RISMapNet(cond_dim=cond_dim, cfg=cfg)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # cond = [x_norm, y_norm, sin(theta), cos(theta)]  -> cond_dim = 4
    model = make_model_preset(cond_dim=4, preset="medium")

    x = torch.randn(8, 3, 256, 256)
    cond = torch.randn(8, 4)

    y = model(x, cond)
    print(y.shape)  # torch.Size([8, 1, 10, 10])
