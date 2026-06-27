from core.schema_rules_swin import infer_swin_defaults

def refine_swin_schema(raw_schema):
    defaults = infer_swin_defaults(raw_schema)
    embed_dim = raw_schema.get("embed_dim") or defaults["embed_dim"]
    depths = raw_schema.get("depths") or defaults["depths"]
    num_heads = raw_schema.get("num_heads") or defaults["num_heads"]
    return {
        "embed_dim": embed_dim, "depths": depths,
        "num_heads": num_heads, "window_size": 7,
        "patch_size": 4, "in_channels": 3,
        "num_classes": raw_schema.get("output", {}).get("num_classes") or 1000
    }
