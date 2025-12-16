# src/generate_code_ready_schema_unet.py

import json
from pathlib import Path

from src.schema_refiner_unet import refine_unet_schema

IN_DIR = Path("outputs/modelspecs")
OUT_DIR = Path("outputs/code_ready_unet")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    for file in IN_DIR.glob("*.json"):
        print(f"Refining schema: {file.name}")

        raw_schema = json.loads(file.read_text(encoding="utf-8"))
        family = raw_schema.get("model_family")

        if family != "unet":
            print(f"⚠️ Skipping unsupported model family: {family}")
            continue   # ✅ now correctly inside loop

        refined = refine_unet_schema(raw_schema)

        out_file = OUT_DIR / file.name
        out_file.write_text(json.dumps(refined, indent=2), encoding="utf-8")

        print(f"  Saved → {out_file}")


if __name__ == "__main__":
    main()
