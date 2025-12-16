import torch
import torch.nn as nn

from src.blocks_vit import PatchEmbedding
from src.blocks_transformer import TransformerEncoderBlock


class ViTBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()

        stem = schema["stem"]["params"]
        block = schema["block"]["params"]

        image_channels = stem["in_channels"]
        patch_size = stem["patch_size"]
        embed_dim = stem["embed_dim"]
        num_patches = stem["num_patches"]

        num_classes = schema["output"]["num_classes"]

        # ---- Patch embedding ----
        self.patch_embed = PatchEmbedding(
            in_channels=image_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        # ---- CLS token ----
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # ---- Positional embedding ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(block.get("dropout", 0.1))

        # ---- Transformer encoder ----
        layers = []
        for stage in schema["stages"]:
            for _ in range(stage["repeats"]):
                layers.append(
                    TransformerEncoderBlock(
                        d_model=block["d_model"],
                        num_heads=block["num_heads"],
                        ffn_dim=block["ffn_dim"],
                        dropout=block.get("dropout", 0.1)
                    )
                )

        self.encoder = nn.Sequential(*layers)

        # ---- Classification head ----
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        B = x.size(0)

        # Patchify
        x = self.patch_embed(x)          # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)

        # Classification via CLS token
        x = self.norm(x[:, 0])
        return self.head(x)
