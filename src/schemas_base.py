# src/schemas_base.py

BASE_MODEL_SCHEMA = {
    "model_family": None,      # resnet | unet | transformer
    "variant": None,           # 50, 101, base, etc.
    "task": None,              # classification | segmentation | nlp

    "input": {
        "channels": None,
        "spatial_dims": None   # 2=image, 1=text
    },

    "output": {
        "num_classes": None,
        "activation": None     # softmax | sigmoid | none
    },

    "stem": {
        "type": None,          # conv | patch_embed | embedding
        "params": {}
    },

    "block": {
        "type": None,          # basic | bottleneck | transformer
        "params": {}
    },

    "stages": [],              # MUST become executable later

    "head": {
        "type": None,
        "params": {}
    }
}
