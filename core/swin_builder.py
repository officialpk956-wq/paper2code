import torch
import torch.nn as nn

from core.blocks_swin import PatchMerging, SwinBlock


class SwinBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        embed_dim = schema["embed_dim"]
        patch_size = schema["patch_size"]
        window_size = schema["window_size"]
        depths = schema["depths"]
        num_heads = schema["num_heads"]
        in_channels = schema["in_channels"]
        num_classes = schema["num_classes"]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            # LayerNorm needs (B, L, C) or (B, H, W, C) so we put it after permute, or use a custom LayerNorm2d.
            # Here we apply it after permute in forward, so we just define LayerNorm.
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        self.stages = nn.ModuleList()
        dim = embed_dim
        for i_layer in range(4):
            stage = nn.ModuleList()
            # Add SwinBlocks
            blocks = nn.ModuleList()
            for i in range(depths[i_layer]):
                shift_size = 0 if (i % 2 == 0) else window_size // 2
                blocks.append(
                    SwinBlock(
                        dim=dim,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=shift_size,
                    )
                )
            stage.append(blocks)

            # Add PatchMerging except for the last stage
            if i_layer < 3:
                stage.append(PatchMerging(dim))
                dim = dim * 2
            else:
                stage.append(nn.Identity())

            self.stages.append(stage)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch Embed
        x = self.patch_embed(x)
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.patch_norm(x)

        for blocks, downsample in self.stages:
            for blk in blocks:
                x = blk(x)
            x = downsample(x)

        x = self.norm(x)
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        # AdaptiveAvgPool
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
