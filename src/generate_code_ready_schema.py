# src/generate_code_ready_schema.py

import json
from pathlib import Path
from src.schema_refiner import refine_resnet_schema

INPUT_DIR = Path("outputs/modelspecs")
OUTPUT_DIR = Path("outputs/code_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    for json_file in INPUT_DIR.glob("*.json"):
        print(f"Refining schema: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            raw_schema = json.load(f)

        family = raw_schema.get("model_family")

        if family == "resnet":
            refined = refine_resnet_schema(raw_schema)
        else:
            print(f"⚠️ Skipping unsupported model family: {family}")
            continue

        out_path = OUTPUT_DIR / json_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(refined, f, indent=2)

        print(f"  Saved → {out_path}")

if __name__ == "__main__":
    main()
