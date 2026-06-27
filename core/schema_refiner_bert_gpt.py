from copy import deepcopy
from core.schema_rules_bert_gpt import infer_bert_gpt_defaults

def refine_bert_gpt_schema(raw_schema):
    schema = deepcopy(raw_schema)
    defaults = infer_bert_gpt_defaults(schema)
    schema["d_model"] = schema.get("d_model") or defaults["d_model"]
    schema["num_heads"] = schema.get("num_heads") or defaults["num_heads"]
    schema["depth"] = schema.get("depth") or defaults["depth"]
    schema["ffn_dim"] = schema.get("ffn_dim") or defaults["ffn_dim"]
    schema["is_causal"] = schema.get("is_causal", defaults["is_causal"])
    schema["vocab_size"] = schema.get("vocab_size", 50257)
    schema["max_seq_len"] = schema.get("max_seq_len", 1024)
    return schema
