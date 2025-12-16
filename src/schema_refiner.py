# src/schema_refiner.py

from copy import deepcopy
from src.schema_rules_resnet import infer_resnet_stage

# Canonical ResNet-50 stage layout
RESNET50_STAGES = [
    {"name": "conv2_x", "repeats": 3, "out_channels": 64},
    {"name": "conv3_x", "repeats": 4, "out_channels": 128},
    {"name": "conv4_x", "repeats": 6, "out_channels": 256},
    {"name": "conv5_x", "repeats": 3, "out_channels": 512},
]

def refine_resnet_schema(raw_schema: dict) -> dict:
    schema = deepcopy(raw_schema)

    # 🔒 Enforce valid ResNet block
    if schema["block"]["type"] not in ("basic", "bottleneck"):
        schema["block"]["type"] = "bottleneck"

    # 🔒 Enforce canonical ResNet stages (do NOT trust LLM here)
    schema["stages"] = RESNET50_STAGES

    refined_stages = []
    prev_channels = schema["stem"]["params"]["out_channels"]

    for idx, stage in enumerate(schema["stages"]):
        inferred = infer_resnet_stage(
            prev_channels=prev_channels,
            stage_out_channels=stage["out_channels"],
            block_type=schema["block"]["type"],
            stage_index=idx
        )

        refined_stage = {
            "name": stage["name"],
            "num_blocks": stage["repeats"],
            **inferred
        }

        refined_stages.append(refined_stage)
        prev_channels = inferred["out_channels"] * inferred["expansion"]

    schema["stages"] = refined_stages
    return schema
