from copy import deepcopy


def refine_vit_schema(raw_schema: dict) -> dict:
    schema = deepcopy(raw_schema)

    # ---- Vision defaults ----
    image_size = schema.get("input", {}).get("spatial_dims") or [224, 224]
    in_channels = schema.get("input", {}).get("channels") or 3

    patch_size = 16
    embed_dim = (
        schema.get("block", {})
        .get("params", {})
        .get("d_model")
        or 768
    )

    num_patches = (image_size[0] // patch_size) ** 2

    # ---- Force ViT stem ----
    schema["stem"] = {
        "type": "patch_embedding",
        "params": {
            "in_channels": in_channels,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "num_patches": num_patches
        }
    }

    # ---- Transformer block defaults ----
    schema["block"] = {
        "type": "transformer_encoder",
        "params": {
            "d_model": embed_dim,
            "num_heads": 12,
            "ffn_dim": embed_dim * 4,
            "dropout": 0.1
        }
    }
        # ---- Transformer block defaults (ViT-safe) ----
    d_model = embed_dim

    # enforce valid head count
    if d_model % 12 == 0:
        num_heads = 12
    elif d_model % 8 == 0:
        num_heads = 8
    else:
        num_heads = 4

    schema["block"] = {
        "type": "transformer_encoder",
        "params": {
            "d_model": d_model,
            "num_heads": num_heads,
            "ffn_dim": d_model * 4,
            "dropout": 0.1
        }
    }

    # ---- Encoder depth (ViT-Base default) ----
    if not schema.get("stages"):
        schema["stages"] = [{"repeats": 12}]

    # ---- Output ----
    schema["output"]["num_classes"] = (
        schema.get("output", {}).get("num_classes") or 1000
    )

    return schema
