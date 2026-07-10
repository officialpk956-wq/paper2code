import torch.nn as nn

from core.blocks_gan import DCGANDiscriminator, DCGANGenerator


class GANBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        latent_dim = schema["latent_dim"]
        base_ch = schema["base_ch"]
        out_ch = schema["img_channels"]

        self.generator = DCGANGenerator(latent_dim=latent_dim, base_ch=base_ch, out_ch=out_ch)
        self.discriminator = DCGANDiscriminator(in_ch=out_ch, base_ch=base_ch)

    def forward(self, z):
        return self.generator(z)
