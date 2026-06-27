from copy import deepcopy
from core.schema_rules_densenet import infer_densenet_defaults

def refine_densenet_schema(raw_schema):
    schema = deepcopy(raw_schema)
    defaults = infer_densenet_defaults(schema)
    schema["num_blocks"] = schema.get("num_blocks") or defaults["num_blocks"]
    schema["growth_rate"] = schema.get("growth_rate") or defaults["growth_rate"]
    schema["init_ch"] = schema.get("init_ch") or defaults["init_ch"]
    schema["num_classes"] = schema.get("output", {}).get("num_classes") or 1000
    return schema
