def infer_mae_defaults(schema):
    return {
        "patch_size": 16,
        "image_size": 224,
        "embed_dim": 768,
        "encoder_depth": 12,
        "encoder_heads": 12,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "mask_ratio": 0.75,
    }
