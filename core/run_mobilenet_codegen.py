import torch
from core.schema_refiner_mobilenet import refine_mobilenet_schema
from core.mobilenet_builder import MobileNetBuilder

def main():
    print("Testing MobileNetV3...")
    schema = {"model_family": "mobilenet", "version": "v3"}
    refined = refine_mobilenet_schema(schema)
    model = MobileNetBuilder(refined)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()
