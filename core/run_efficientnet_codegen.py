import torch
from core.schema_refiner_efficientnet import refine_efficientnet_schema
from core.efficientnet_builder import EfficientNetBuilder

def main():
    schema = {"model_family": "efficientnet", "variant": "b0"}
    refined = refine_efficientnet_schema(schema)
    model = EfficientNetBuilder(refined)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
