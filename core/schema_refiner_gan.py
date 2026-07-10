from core.schema_rules_gan import infer_gan_defaults


def refine_gan_schema(raw_schema):
    defaults = infer_gan_defaults(raw_schema)
    schema = {}
    schema["latent_dim"] = raw_schema.get("latent_dim") or defaults["latent_dim"]
    schema["base_ch"] = raw_schema.get("base_ch") or defaults["base_ch"]
    schema["img_channels"] = raw_schema.get("img_channels") or defaults["img_channels"]
    schema["image_size"] = raw_schema.get("image_size") or defaults["image_size"]
    return schema
