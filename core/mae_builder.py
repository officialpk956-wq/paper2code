import torch
import torch.nn as nn

from core.blocks_mae import MAEDecoder, RandomMasking
from core.blocks_transformer import TransformerEncoderBlock
from core.blocks_vit import PatchEmbedding


class MAEBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        patch_size = schema["patch_size"]
        image_size = schema["image_size"]
        embed_dim = schema["embed_dim"]
        encoder_depth = schema["encoder_depth"]
        encoder_heads = schema["encoder_heads"]
        decoder_dim = schema["decoder_dim"]
        decoder_depth = schema["decoder_depth"]
        decoder_heads = schema["decoder_heads"]
        mask_ratio = schema["mask_ratio"]

        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.masking = RandomMasking(mask_ratio)

        encoder_blocks = []
        for _ in range(encoder_depth):
            encoder_blocks.append(
                TransformerEncoderBlock(
                    d_model=embed_dim, num_heads=encoder_heads, ffn_dim=embed_dim * 4
                )
            )
        self.encoder = nn.Sequential(*encoder_blocks)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))

        self.decoder = MAEDecoder(
            encoder_dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            patch_size=patch_size,
        )

    def forward(self, x, mask=True):
        x = self.patch_embed(x)
        B, N, D = x.shape

        x = x + self.pos_embed[:, 1:, :]

        if mask:
            x, mask_indices, ids_restore = self.masking(x)
        else:
            mask_indices = None
            ids_restore = None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)

        # Decoder forward logic
        x = self.decoder.proj(x)

        if mask:
            mask_tokens = self.mask_token.repeat(B, N + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder.blocks:
            x = blk(x)
        x = self.decoder.norm(x)

        # Remove cls token
        x = x[:, 1:, :]

        x = self.decoder.head(x)
        return x
