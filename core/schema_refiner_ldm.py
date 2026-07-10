from core.schema_rules_ldm import infer_ldm_defaults


def refine_ldm_schema(raw_schema):
    defaults = infer_ldm_defaults(raw_schema)
    schema = {}
    schema["in_channels"] = raw_schema.get("in_channels") or defaults["in_channels"]
    schema["out_channels"] = raw_schema.get("out_channels") or defaults["out_channels"]
    schema["model_channels"] = raw_schema.get("model_channels") or defaults["model_channels"]
    schema["attention_resolutions"] = (
        raw_schema.get("attention_resolutions") or defaults["attention_resolutions"]
    )
    schema["num_res_blocks"] = raw_schema.get("num_res_blocks") or defaults["num_res_blocks"]
    schema["channel_mult"] = raw_schema.get("channel_mult") or defaults["channel_mult"]
    schema["num_heads"] = raw_schema.get("num_heads") or defaults["num_heads"]
    schema["context_dim"] = raw_schema.get("context_dim") or defaults["context_dim"]
    return schema
