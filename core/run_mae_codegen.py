import torch
from core.schema_refiner_mae import refine_mae_schema
from core.mae_builder import MAEBuilder

def main():
    schema = {"model_family": "mae"}
    refined = refine_mae_schema(schema)
    model = MAEBuilder(refined)
    x = torch.randn(1, 3, 224, 224)
    out = model(x, mask=True)
    print(f"Output shape (masked): {out.shape}")
    out_unmasked = model(x, mask=False)
    print(f"Output shape (unmasked): {out_unmasked.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
