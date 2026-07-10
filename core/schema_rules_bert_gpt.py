def infer_bert_gpt_defaults(schema):
    style = schema.get("style", "bert")  # "bert" or "gpt"
    CONFIGS = {
        "bert-base": {
            "d_model": 768,
            "num_heads": 12,
            "depth": 12,
            "ffn_dim": 3072,
            "is_causal": False,
        },
        "bert-large": {
            "d_model": 1024,
            "num_heads": 16,
            "depth": 24,
            "ffn_dim": 4096,
            "is_causal": False,
        },
        "gpt2": {"d_model": 768, "num_heads": 12, "depth": 12, "ffn_dim": 3072, "is_causal": True},
        "gpt2-large": {
            "d_model": 1280,
            "num_heads": 20,
            "depth": 36,
            "ffn_dim": 5120,
            "is_causal": True,
        },
        "bert": {"d_model": 768, "num_heads": 12, "depth": 12, "ffn_dim": 3072, "is_causal": False},
        "gpt": {"d_model": 768, "num_heads": 12, "depth": 12, "ffn_dim": 3072, "is_causal": True},
    }
    return CONFIGS.get(schema.get("variant", style), CONFIGS["bert"])
