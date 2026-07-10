from copy import deepcopy

from core.schema_rules_mobilenet import infer_mobilenet_defaults


def refine_mobilenet_schema(raw_schema):
    schema = deepcopy(raw_schema)
    defaults = infer_mobilenet_defaults(schema)
    schema["version"] = schema.get("version") or defaults["version"]
    schema["multiplier"] = schema.get("multiplier") or defaults["multiplier"]
    schema["stages"] = schema.get("stages") or defaults["stages"]
    schema["num_classes"] = schema.get("output", {}).get("num_classes") or 1000
    return schema
