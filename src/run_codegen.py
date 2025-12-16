import json
import torch
from src.model_builder import ResNetBuilder

schema = json.load(open("outputs/code_ready/resnet_he_2015.json"))
model = ResNetBuilder(schema)

x = torch.randn(1, 3, 224, 224)
y = model(x)

print("Output shape:", y.shape)
