def infer_vit_defaults(schema: dict) -> dict:
    """
    Fill safe defaults for Vision Transformer models.
    """

    variant = schema.get("variant") or "base"

    if variant == "base":
        return {
            "embed_dim": 768,
            "num_heads": 12,
            "depth": 12,
            "ffn_dim": 3072,
            "patch_size": 16
        }

    if variant == "large":
        return {
            "embed_dim": 1024,
            "num_heads": 16,
            "depth": 24,
            "ffn_dim": 4096,
            "patch_size": 16
        }

    # fallback
    return {
        "embed_dim": 768,
        "num_heads": 12,
        "depth": 12,
        "ffn_dim": 3072,
        "patch_size": 16
    }
