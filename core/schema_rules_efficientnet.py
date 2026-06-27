def infer_efficientnet_defaults(schema):
    variant = schema.get("variant", "b0")
    # phi=0: width=1.0, depth=1.0; b1: w=1.1,d=1.1; b3: w=1.2,d=1.4
    CONFIGS = {
        "b0": {"width_coeff": 1.0, "depth_coeff": 1.0, "image_size": 224},
        "b1": {"width_coeff": 1.1, "depth_coeff": 1.1, "image_size": 240},
        "b3": {"width_coeff": 1.2, "depth_coeff": 1.4, "image_size": 300},
        "b7": {"width_coeff": 2.0, "depth_coeff": 3.1, "image_size": 600},
    }
    return CONFIGS.get(variant, CONFIGS["b0"])
