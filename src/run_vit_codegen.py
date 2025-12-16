import json
import torch

from src.vit_builder import ViTBuilder

schema = json.load(
    open("outputs/code_ready/attention_all_you_need_2017.json")
)

model = ViTBuilder(schema)

x = torch.randn(2, 3, 224, 224)
y = model(x)

print("Output shape:", y.shape)
