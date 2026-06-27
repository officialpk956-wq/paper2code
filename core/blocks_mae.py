import torch
import torch.nn as nn
from core.blocks_transformer import TransformerEncoderBlock

class RandomMasking(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Gather visible patches
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore

class MAEDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, depth, num_heads, patch_size):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        
        blocks = []
        for _ in range(depth):
            blocks.append(TransformerEncoderBlock(d_model=decoder_dim, num_heads=num_heads, ffn_dim=decoder_dim * 4))
        self.blocks = nn.Sequential(*blocks)
        
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size**2 * 3)

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x
