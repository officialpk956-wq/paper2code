def infer_swin_defaults(schema):
    variant = schema.get("variant", "tiny")
    CONFIGS = {
        "tiny": {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
        "small": {"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]},
        "base": {"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]},
        "large": {"embed_dim": 192, "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48]},
    }
    return CONFIGS.get(variant, CONFIGS["tiny"])
