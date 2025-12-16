# src/unet_builder.py

import torch
import torch.nn as nn
from src.blocks_unet import DoubleConv

class UNetBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        in_ch = schema["input"]["channels"]

        # Encoder
        for ch in schema["encoder"]:
            self.encoders.append(DoubleConv(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, schema["bottleneck"])

        # Decoder
        in_ch = schema["bottleneck"]
        for ch in schema["decoder"]:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, ch, 2, stride=2))
            self.decoders.append(DoubleConv(in_ch, ch))
            in_ch = ch

        self.final = nn.Conv2d(
            in_ch,
            schema["output"]["num_classes"] or 1,
            kernel_size=1
        )

    def forward(self, x):
        skips = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(
            self.upconvs, self.decoders, reversed(skips)
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final(x)
