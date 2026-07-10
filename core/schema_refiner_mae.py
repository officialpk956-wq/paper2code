from core.schema_rules_mae import infer_mae_defaults


def refine_mae_schema(raw_schema):
    defaults = infer_mae_defaults(raw_schema)
    schema = {}
    schema["patch_size"] = raw_schema.get("patch_size") or defaults["patch_size"]
    schema["image_size"] = raw_schema.get("image_size") or defaults["image_size"]
    schema["embed_dim"] = raw_schema.get("embed_dim") or defaults["embed_dim"]
    schema["encoder_depth"] = raw_schema.get("encoder_depth") or defaults["encoder_depth"]
    schema["encoder_heads"] = raw_schema.get("encoder_heads") or defaults["encoder_heads"]
    schema["decoder_dim"] = raw_schema.get("decoder_dim") or defaults["decoder_dim"]
    schema["decoder_depth"] = raw_schema.get("decoder_depth") or defaults["decoder_depth"]
    schema["decoder_heads"] = raw_schema.get("decoder_heads") or defaults["decoder_heads"]
    schema["mask_ratio"] = raw_schema.get("mask_ratio") or defaults["mask_ratio"]
    return schema
