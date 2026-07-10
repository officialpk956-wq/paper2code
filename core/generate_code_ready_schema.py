# src/generate_code_ready_schema.py

import json
from pathlib import Path

from core.schema_refiner import refine_resnet_schema

INPUT_DIR = Path("outputs/modelspecs")
OUTPUT_DIR = Path("outputs/code_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    for json_file in INPUT_DIR.glob("*.json"):
        print(f"Refining schema: {json_file.name}")

        with open(json_file, encoding="utf-8") as f:
            raw_schema = json.load(f)

        family = raw_schema.get("model_family")

        if family == "resnet":
            refined = refine_resnet_schema(raw_schema)
        elif family == "unet":
            from core.schema_refiner_unet import refine_unet_schema

            refined = refine_unet_schema(raw_schema)
        elif family == "transformer":
            from core.schema_refiner_transformer import refine_transformer_schema

            refined = refine_transformer_schema(raw_schema)
        elif family == "vit":
            from core.schema_refiner_vit import refine_vit_schema

            refined = refine_vit_schema(raw_schema)
        elif family == "efficientnet":
            from core.schema_refiner_efficientnet import refine_efficientnet_schema

            refined = refine_efficientnet_schema(raw_schema)
        elif family == "swin":
            from core.schema_refiner_swin import refine_swin_schema

            refined = refine_swin_schema(raw_schema)
        elif family == "gan":
            from core.schema_refiner_gan import refine_gan_schema

            refined = refine_gan_schema(raw_schema)
        elif family == "densenet":
            from core.schema_refiner_densenet import refine_densenet_schema

            refined = refine_densenet_schema(raw_schema)
        elif family == "bert_gpt":
            from core.schema_refiner_bert_gpt import refine_bert_gpt_schema

            refined = refine_bert_gpt_schema(raw_schema)
        elif family == "mobilenet":
            from core.schema_refiner_mobilenet import refine_mobilenet_schema

            refined = refine_mobilenet_schema(raw_schema)
        elif family == "mae":
            from core.schema_refiner_mae import refine_mae_schema

            refined = refine_mae_schema(raw_schema)
        elif family == "ldm":
            from core.schema_refiner_ldm import refine_ldm_schema

            refined = refine_ldm_schema(raw_schema)
        elif family in ("diffusion", "yolo"):
            refined = raw_schema  # No refiner currently defined for these
        else:
            print(f"⚠️ Skipping unsupported model family: {family}")
            continue

        out_path = OUTPUT_DIR / json_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(refined, f, indent=2)

        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
