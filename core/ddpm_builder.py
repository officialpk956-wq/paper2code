import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Time embedding for Diffusion Models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """ResNet Block with Time Embedding projection."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        h = self.block1(x)
        # Add time embedding broadcasted across spatial dimensions
        time_emb = self.mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class DDPMBuilder(nn.Module):
    """
    DDPM (Denoising Diffusion Probabilistic Model) UNet Architecture.
    This is an executable PyTorch blueprint.
    """

    def __init__(self, channels: int = 3, dim: int = 64):
        super().__init__()
        self.channels = channels

        # Time Embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial projection
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)

        # Downsample path
        self.down1 = ResnetBlock(dim, dim, time_dim)
        self.down2 = ResnetBlock(dim, dim * 2, time_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ResnetBlock(dim * 2, dim * 2, time_dim)
        self.mid2 = ResnetBlock(dim * 2, dim * 2, time_dim)

        # Upsample path
        self.up1 = ResnetBlock(
            dim * 4, dim, time_dim
        )  # *4 because of concatenation (dim*2 from down + dim*2 from mid)
        self.up2 = ResnetBlock(dim * 2, dim, time_dim)  # *2 because of concatenation

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Output projection
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time):
        # 1. Embed time
        t = self.time_mlp(time)

        # 2. Initial convolution
        x = self.init_conv(x)

        # 3. Downsampling (Encoder)
        d1 = self.down1(x, t)
        d2 = self.down2(self.pool(d1), t)

        # 4. Bottleneck
        m = self.mid1(d2, t)
        m = self.mid2(m, t)

        # 5. Upsampling (Decoder) with skip connections
        u1 = self.up1(torch.cat([m, d2], dim=1), t)
        u2 = self.up2(torch.cat([self.upsample(u1), d1], dim=1), t)

        # 6. Output noise prediction
        out = self.final_conv(u2)
        return out
