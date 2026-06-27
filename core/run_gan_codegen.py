import torch
from core.schema_refiner_gan import refine_gan_schema
from core.gan_builder import GANBuilder

def main():
    schema = {"model_family": "gan", "latent_dim": 100}
    refined = refine_gan_schema(schema)
    model = GANBuilder(refined)
    z = torch.randn(4, 100)
    out = model(z)
    print(f"Generator output shape: {out.shape}")
    print(f"Generator params: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in model.discriminator.parameters()):,}")

if __name__ == "__main__":
    main()
