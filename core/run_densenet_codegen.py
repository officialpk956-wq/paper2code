import torch
from core.schema_refiner_densenet import refine_densenet_schema
from core.densenet_builder import DenseNetBuilder

def main():
    schema = {"model_family": "densenet", "variant": "121"}
    refined = refine_densenet_schema(schema)
    model = DenseNetBuilder(refined)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
