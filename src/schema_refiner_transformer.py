# src/schema_refiner_transformer.py

from copy import deepcopy
from src.schema_rules_transformer import infer_transformer_block


def refine_transformer_schema(raw_schema: dict) -> dict:
    """
    Convert a normalized Transformer schema into a code-ready schema.
    """

    schema = deepcopy(raw_schema)

    # Global block parameters (Transformer blocks are identical)
    block_params = infer_transformer_block(
        schema["block"]["params"]
    )

    refined_stages = []

    for stage in schema["stages"]:
        refined_stages.append({
            "num_blocks": stage["repeats"],
            "block": block_params
        })

    schema["stages"] = refined_stages
    return schema
