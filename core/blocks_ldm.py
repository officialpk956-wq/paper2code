import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(t_emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class SpatialTransformer(nn.Module):
    def __init__(self, in_ch, n_heads, d_head, context_dim=None):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.proj_in = nn.Conv2d(in_ch, n_heads * d_head, 1)

        self.n_heads = n_heads
        self.d_head = d_head

        self.attn1 = nn.MultiheadAttention(n_heads * d_head, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(n_heads * d_head)

        self.context_dim = context_dim
        if context_dim is not None:
            self.attn2 = nn.MultiheadAttention(
                n_heads * d_head, n_heads, batch_first=True, kdim=context_dim, vdim=context_dim
            )
            self.norm2 = nn.LayerNorm(n_heads * d_head)

        self.ff = nn.Sequential(
            nn.Linear(n_heads * d_head, n_heads * d_head * 4),
            nn.GELU(),
            nn.Linear(n_heads * d_head * 4, n_heads * d_head),
        )
        self.norm3 = nn.LayerNorm(n_heads * d_head)

        self.proj_out = nn.Conv2d(n_heads * d_head, in_ch, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Self attn
        x = x + self.attn1(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Cross attn
        if self.context_dim is not None and context is not None:
            x = x + self.attn2(self.norm2(x), context, context)[0]

        # FF
        x = x + self.ff(self.norm3(x))

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x_in + x
