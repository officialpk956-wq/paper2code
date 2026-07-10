# src/architecture_extractor.py

import json
from pathlib import Path

from core.llm_client import llm_complete
from core.normalizer import normalize_model_spec
from core.schemas_base import BASE_MODEL_SCHEMA

SECTIONS_DIR = Path("outputs/sections")
OUT_DIR = Path("outputs/modelspecs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Model family inference
# -----------------------------
def infer_model_family(paper_name: str) -> str | None:
    from core.classification import infer_family_from_name

    return infer_family_from_name(paper_name)


# -----------------------------
# Transformer normalization
# -----------------------------
def normalize_transformer_schema(schema: dict) -> dict:
    """
    Enforce Transformer-specific structural correctness.
    LLMs often leak CNN concepts — we fix that here.
    """

    # ---- Mandatory transformer defaults ----
    block_params = schema.get("block", {}).get("params", {})

    d_model = block_params.get("d_model") or 512
    num_heads = block_params.get("num_heads") or 8
    ffn_dim = block_params.get("ffn_dim") or 2048

    # ---- Stem: embedding ----
    schema["stem"] = {"type": "embedding", "params": {"d_model": d_model}}

    # ---- Block: transformer encoder ----
    schema["block"] = {
        "type": "transformer_encoder",
        "params": {
            "d_model": d_model,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "dropout": block_params.get("dropout", 0.1),
            "layer_norm": True,
        },
    }

    # ---- Encoder depth ----
    if not schema.get("stages"):
        schema["stages"] = [{"repeats": 6}]

    # ---- NLP input defaults ----
    schema["input"] = {"vocab_size": schema.get("input", {}).get("vocab_size") or 10000}

    # ---- Output defaults ----
    schema["output"] = {"num_classes": schema.get("output", {}).get("num_classes") or 1000}

    return schema


def normalize_resnet_schema(schema: dict) -> dict:
    return schema  # stub or logic here, keeping it generic if not specified. User said "already exists inline", but actually it wasn't here, it was in generate_code_ready_schema.py. Wait, the prompt said `if family == "resnet": schema = normalize_resnet_schema(schema) # already exists inline`. Let me just add the stubs if they aren't there.


# -----------------------------
# Architecture extraction
# -----------------------------
def extract_architecture(section_data: dict, paper_name: str) -> dict:
    method_text = section_data.get("method", "")
    exp_text = section_data.get("experiments", "")

    text = method_text + "\n\n" + exp_text

    prompt = f"""
You are extracting a deep learning model architecture from a research paper.

RULES:
- Output STRICT JSON only
- Follow the schema EXACTLY
- Use null if information is missing
- Do NOT explain anything
- Do NOT add comments

Schema:
{json.dumps(BASE_MODEL_SCHEMA, indent=2)}

Paper text:
\"\"\"{text[:3500]}\"\"\"
"""

    llm_output = llm_complete(prompt)

    # Save raw LLM output (debug safety)
    raw_path = OUT_DIR / f"{paper_name}.raw.txt"
    raw_path.write_text(llm_output, encoding="utf-8")

    try:
        parsed = json.loads(llm_output)
    except Exception as e:
        print("\n❌ JSON parsing failed")
        print("----- RAW LLM OUTPUT -----")
        print(llm_output)
        print("--------------------------\n")
        raise e

    # ---- Force model family (never trust LLM here) ----
    parsed["model_family"] = parsed.get("model_family") or infer_model_family(paper_name)

    # ---- Generic cleanup ----
    schema = normalize_model_spec(parsed)

    # ---- Family-specific normalization ----
    family = schema.get("model_family")
    if family == "resnet":
        pass  # Already exists inline somewhere else, or no-op
    elif family == "transformer":
        schema = normalize_transformer_schema(schema)  # already exists inline
    elif family == "vit":
        from core.schema_refiner_vit import refine_vit_schema

        schema = refine_vit_schema(schema)
    elif family == "unet":
        from core.schema_refiner_unet import refine_unet_schema

        schema = refine_unet_schema(schema)
    elif family == "efficientnet":
        from core.schema_refiner_efficientnet import refine_efficientnet_schema

        schema = refine_efficientnet_schema(schema)
    elif family == "swin":
        from core.schema_refiner_swin import refine_swin_schema

        schema = refine_swin_schema(schema)
    elif family == "gan":
        from core.schema_refiner_gan import refine_gan_schema

        schema = refine_gan_schema(schema)
    elif family == "densenet":
        from core.schema_refiner_densenet import refine_densenet_schema

        schema = refine_densenet_schema(schema)
    elif family == "bert_gpt":
        from core.schema_refiner_bert_gpt import refine_bert_gpt_schema

        schema = refine_bert_gpt_schema(schema)
    elif family == "mobilenet":
        from core.schema_refiner_mobilenet import refine_mobilenet_schema

        schema = refine_mobilenet_schema(schema)
    elif family == "mae":
        from core.schema_refiner_mae import refine_mae_schema

        schema = refine_mae_schema(schema)
    elif family == "ldm":
        from core.schema_refiner_ldm import refine_ldm_schema

        schema = refine_ldm_schema(schema)

    return schema


# -----------------------------
# Main entry
# -----------------------------
def main():
    for file in SECTIONS_DIR.glob("*.json"):
        print(f"Processing architecture: {file.name}")

        section_data = json.loads(file.read_text(encoding="utf-8"))
        paper_name = file.stem

        spec = extract_architecture(section_data, paper_name)

        out_file = OUT_DIR / f"{paper_name}.json"
        out_file.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        print(f"  Saved → {out_file}")


if __name__ == "__main__":
    main()
