import json
from pathlib import Path

from src.diagram_resnet import draw_resnet
from src.diagram_unet import draw_unet
from src.diagram_vit import draw_vit




SCHEMA_DIR = Path("outputs/code_ready")
OUT_DIR = Path("outputs/diagrams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    for file in SCHEMA_DIR.glob("*.json"):
        schema = json.loads(file.read_text())
        family = schema["model_family"]

        if family == "resnet":
            graph = draw_resnet(schema)
        elif family == "unet":
            graph = draw_unet(schema)
        elif family == "transformer":
            graph = draw_vit(schema)
        else:
            continue

        graph.render(OUT_DIR / file.stem, format="png")
        print(f"Saved diagram → {file.stem}.png")

if __name__ == "__main__":
    main()
