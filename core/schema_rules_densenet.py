def infer_densenet_defaults(schema):
    variant = schema.get("variant", "121")
    CONFIGS = {
        "121": {"num_blocks": [6, 12, 24, 16], "growth_rate": 32, "init_ch": 64},
        "169": {"num_blocks": [6, 12, 32, 32], "growth_rate": 32, "init_ch": 64},
        "201": {"num_blocks": [6, 12, 48, 32], "growth_rate": 32, "init_ch": 64},
    }
    return CONFIGS.get(variant, CONFIGS["121"])
