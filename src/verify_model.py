import json
import torch

from src.param_counter import count_parameters
from src.flops_estimator import (
    estimate_vit_flops,
    estimate_resnet_flops,
    estimate_unet_flops,
)

from src.vit_builder import ViTBuilder
from src.model_builder import ResNetBuilder
from src.unet_builder import UNetBuilder


def verify(schema_path: str):
    schema = json.load(open(schema_path))
    family = schema["model_family"]

    if family == "transformer":
        model = ViTBuilder(schema)
        flops = estimate_vit_flops(schema)

    elif family == "resnet":
        model = ResNetBuilder(schema)
        flops = estimate_resnet_flops(schema)

    elif family == "unet":
        model = UNetBuilder(schema)
        flops = estimate_unet_flops(schema)

    else:
        raise ValueError(f"Unsupported model family: {family}")

    params = count_parameters(model)

    print(f"\nModel: {schema_path}")
    print(f"Parameters: {params:,}")
    print(f"Estimated FLOPs: {flops:,}")


if __name__ == "__main__":
    verify("outputs/code_ready/attention_all_you_need_2017.json")
