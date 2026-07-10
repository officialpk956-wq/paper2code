import torch

from core.ldm_builder import LDMBuilder
from core.schema_refiner_ldm import refine_ldm_schema


def main():
    schema = {"model_family": "ldm"}
    refined = refine_ldm_schema(schema)
    model = LDMBuilder(refined)
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 10, 768)
    out = model(x, t, context)
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
