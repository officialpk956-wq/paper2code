import torch
import torch.nn as nn
from core.blocks_ldm import ResBlock, SpatialTransformer
from core.ddpm_builder import SinusoidalPositionEmbeddings

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class LDMBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        
        in_channels = schema["in_channels"]
        out_channels = schema["out_channels"]
        model_channels = schema["model_channels"]
        attention_resolutions = schema["attention_resolutions"]
        num_res_blocks = schema["num_res_blocks"]
        channel_mult = schema["channel_mult"]
        num_heads = schema["num_heads"]
        context_dim = schema["context_dim"]
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        # Down
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, model_channels * mult, time_embed_dim)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, num_heads, ch // num_heads, context_dim))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
                ds *= 2
                
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim),
            SpatialTransformer(ch, num_heads, ch // num_heads, context_dim),
            ResBlock(ch, ch, time_embed_dim)
        ])
        
        # Up
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, model_channels * mult, time_embed_dim)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, num_heads, ch // num_heads, context_dim))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))
                
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, context=None):
        t_emb = self.time_embed(timesteps)
        
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
            
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            elif isinstance(module, SpatialTransformer):
                h = module(h, context)
            else:
                h = module(h)
                
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, SpatialTransformer):
                    h = layer(h, context)
                else:
                    h = layer(h)
                    
        return self.out(h)
