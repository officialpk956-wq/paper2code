import torch
from core.schema_refiner_swin import refine_swin_schema
from core.swin_builder import SwinBuilder

def main():
    schema = {"model_family": "swin", "variant": "tiny"}
    refined = refine_swin_schema(schema)
    model = SwinBuilder(refined)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
