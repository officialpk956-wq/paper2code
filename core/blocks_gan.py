import torch.nn as nn


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, base_ch=64, out_ch=3):
        super().__init__()
        self.base_ch = base_ch

        self.proj = nn.Linear(latent_dim, base_ch * 8 * 4 * 4)
        self.blocks = nn.Sequential(
            GenBlock(base_ch * 8, base_ch * 4),
            GenBlock(base_ch * 4, base_ch * 2),
            GenBlock(base_ch * 2, base_ch * 1),
            nn.ConvTranspose2d(base_ch * 1, out_ch, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.proj(z)
        x = x.view(-1, self.base_ch * 8, 4, 4)
        return self.blocks(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, base_ch * 1, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DiscBlock(base_ch * 1, base_ch * 2, stride=2),
            DiscBlock(base_ch * 2, base_ch * 4, stride=2),
            DiscBlock(base_ch * 4, base_ch * 8, stride=2),
            nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.blocks(x)
