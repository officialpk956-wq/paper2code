def infer_ldm_defaults(schema):
    return {
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "attention_resolutions": [4, 2, 1],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "num_heads": 8,
        "context_dim": 768
    }
