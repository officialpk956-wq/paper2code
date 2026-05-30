# src/schema_rules_transformer.py

def infer_transformer_block(block_params: dict):
    """
    Fill defaults for a Transformer encoder block
    using standard paper defaults when missing.
    """

    d_model = block_params.get("d_model") or 512
    num_heads = block_params.get("num_heads") or 8
    ffn_dim = block_params.get("ffn_dim") or 2048

    return {
        "d_model": d_model,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "dropout": block_params.get("dropout", 0.1),
        "layer_norm": True
    }
