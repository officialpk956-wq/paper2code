# src/schema_refiner_unet.py

from copy import deepcopy
from src.schema_rules_unet import build_unet_channels

def refine_unet_schema(raw_schema: dict) -> dict:
    schema = deepcopy(raw_schema)

    # 🔒 Ensure input channels (do NOT trust LLM)
    if not schema.get("input"):
        schema["input"] = {}

    if schema["input"].get("channels") is None:
        # Safe default for UNet (medical imaging)
        schema["input"]["channels"] = 1

    # 🔒 Ensure output classes
    if not schema.get("output"):
        schema["output"] = {}

    if schema["output"].get("num_classes") is None:
        schema["output"]["num_classes"] = 1

    # Base channels (default UNet = 64)
    base_channels = (
        schema.get("stem", {})
        .get("params", {})
        .get("out_channels", 64)
    )

    depth = schema.get("depth", 4)

    encoder, bottleneck, decoder = build_unet_channels(
        base_channels=base_channels,
        depth=depth
    )

    schema["encoder"] = encoder
    schema["bottleneck"] = bottleneck
    schema["decoder"] = decoder

    return schema
