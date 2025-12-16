# src/architecture_extractor.py

import json
from pathlib import Path

from src.llm_client import llm_complete
from src.schemas_base import BASE_MODEL_SCHEMA
from src.normalizer import normalize_model_spec

SECTIONS_DIR = Path("outputs/sections")
OUT_DIR = Path("outputs/modelspecs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Model family inference
# -----------------------------
def infer_model_family(paper_name: str) -> str | None:
    name = paper_name.lower()

    if "resnet" in name:
        return "resnet"
    if "unet" in name:
        return "unet"
    if "attention" in name or "transformer" in name:
        return "transformer"

    return None


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
    schema["stem"] = {
        "type": "embedding",
        "params": {
            "d_model": d_model
        }
    }

    # ---- Block: transformer encoder ----
    schema["block"] = {
        "type": "transformer_encoder",
        "params": {
            "d_model": d_model,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "dropout": block_params.get("dropout", 0.1),
            "layer_norm": True
        }
    }

    # ---- Encoder depth ----
    if not schema.get("stages"):
        schema["stages"] = [
            {"repeats": 6}
        ]

    # ---- NLP input defaults ----
    schema["input"] = {
        "vocab_size": schema.get("input", {}).get("vocab_size") or 10000
    }

    # ---- Output defaults ----
    schema["output"] = {
        "num_classes": schema.get("output", {}).get("num_classes") or 1000
    }

    return schema


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
    parsed["model_family"] = (
        parsed.get("model_family")
        or infer_model_family(paper_name)
    )

    # ---- Generic cleanup ----
    schema = normalize_model_spec(parsed)

    # ---- Family-specific normalization ----
    if schema["model_family"] == "transformer":
        schema = normalize_transformer_schema(schema)

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
