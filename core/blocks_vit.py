import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, D)
        """
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2)                 # (B, D, N)
        x = x.transpose(1, 2)            # (B, N, D)
        return x
