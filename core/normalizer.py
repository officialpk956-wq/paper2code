# src/normalizer.py

from copy import deepcopy


def normalize_model_spec(raw: dict) -> dict:
    """
    Normalize LLM output into strict internal schema.
    This is where we fix naming inconsistencies.
    """

    spec = deepcopy(raw) or {}

    # ---- STEM ----
    stem = spec.get("stem") or {}
    params = stem.get("params") or {}

    # Normalize num_filters -> out_channels
    if "out_channels" not in params:
        if "num_filters" in params:
            params["out_channels"] = params.pop("num_filters")

    # Default ResNet stem if missing or explicitly null (setdefault only
    # covers a missing key -- the LLM sometimes returns "stride": null etc.,
    # which setdefault leaves untouched and later crashes tensor_tracker).
    params["out_channels"] = params.get("out_channels") or 64
    params["kernel"] = params.pop("kernel_size", None) or params.get("kernel") or 7
    params["stride"] = params.get("stride") or 2
    params["padding"] = params.get("padding") or 3

    stem["type"] = stem.get("type") or "conv"
    stem["params"] = params
    spec["stem"] = stem

    # ---- BLOCK ----
    block = spec.get("block") or {}
    block["type"] = block.get("type") or "bottleneck"
    if block.get("params") is None:
        block["params"] = {}
    spec["block"] = block

    # ---- STAGES ----
    stages = spec.get("stages") or []
    normalized_stages = []

    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_params = stage.get("params") or {}
        normalized_stages.append(
            {
                "name": stage.get("name"),
                "repeats": stage.get("repeats") or 1,
                "out_channels": stage.get("out_channels")
                or stage_params.get("num_filters"),
            }
        )

    spec["stages"] = normalized_stages

    return spec
