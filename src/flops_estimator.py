def estimate_vit_flops(schema):
    """
    Rough FLOPs estimate for Vision Transformer.
    """
    d = schema["block"]["params"]["d_model"]
    n = schema["stem"]["params"]["num_patches"]
    layers = sum(stage["repeats"] for stage in schema["stages"])

    # Self-attention + FFN (approximate)
    flops_per_layer = 4 * n * d * d
    return layers * flops_per_layer


def estimate_resnet_flops(schema):
    flops = 0
    for stage in schema["stages"]:
        c = stage["out_channels"]
        blocks = stage["num_blocks"]
        flops += blocks * (9 * c * c)  # rough conv estimate
    return flops


def estimate_unet_flops(schema):
    flops = 0
    for stage in schema["encoder"]:
        c = stage["out_channels"]
        flops += 9 * c * c
    for stage in schema["decoder"]:
        c = stage["out_channels"]
        flops += 9 * c * c
    return flops
