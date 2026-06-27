from copy import deepcopy
from core.schema_rules_efficientnet import infer_efficientnet_defaults

def refine_efficientnet_schema(raw_schema):
    schema = deepcopy(raw_schema)
    defaults = infer_efficientnet_defaults(schema)
    schema["width_coeff"] = schema.get("width_coeff") or defaults["width_coeff"]
    schema["depth_coeff"] = schema.get("depth_coeff") or defaults["depth_coeff"]
    schema["num_classes"] = schema.get("output", {}).get("num_classes") or 1000
    # baseline stage config (7 stages, EfficientNet-B0 widths)
    if not schema.get("stages"):
        schema["stages"] = [
            {"expand_ratio": 1, "out_ch": 16,  "num_blocks": 1, "stride": 1},
            {"expand_ratio": 6, "out_ch": 24,  "num_blocks": 2, "stride": 2},
            {"expand_ratio": 6, "out_ch": 40,  "num_blocks": 2, "stride": 2},
            {"expand_ratio": 6, "out_ch": 80,  "num_blocks": 3, "stride": 2},
            {"expand_ratio": 6, "out_ch": 112, "num_blocks": 3, "stride": 1},
            {"expand_ratio": 6, "out_ch": 192, "num_blocks": 4, "stride": 2},
            {"expand_ratio": 6, "out_ch": 320, "num_blocks": 1, "stride": 1},
        ]
    return schema
