# src/normalizer.py

from copy import deepcopy

def normalize_model_spec(raw: dict) -> dict:
    """
    Normalize LLM output into strict internal schema.
    This is where we fix naming inconsistencies.
    """

    spec = deepcopy(raw)

    # ---- STEM ----
    stem = spec.get("stem", {})
    params = stem.get("params", {})

    # Normalize num_filters -> out_channels
    if "out_channels" not in params:
        if "num_filters" in params:
            params["out_channels"] = params.pop("num_filters")

    # Default ResNet stem if missing
    params.setdefault("out_channels", 64)
    params.setdefault("kernel", params.pop("kernel_size", 7))
    params.setdefault("stride", 2)
    params.setdefault("padding", 3)

    stem["type"] = stem.get("type") or "conv"
    stem["params"] = params
    spec["stem"] = stem

    # ---- BLOCK ----
    block = spec.get("block", {})
    block["type"] = block.get("type") or "bottleneck"
    block.setdefault("params", {})
    spec["block"] = block

    # ---- STAGES ----
    stages = spec.get("stages", [])
    normalized_stages = []

    for stage in stages:
        normalized_stages.append({
            "name": stage.get("name"),
            "repeats": stage.get("repeats", 1),
            "out_channels": stage.get("out_channels") or
                            stage.get("params", {}).get("num_filters")
        })

    spec["stages"] = normalized_stages

    return spec

