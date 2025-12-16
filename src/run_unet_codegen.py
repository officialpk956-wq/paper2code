# src/run_unet_codegen.py

import json
import torch
from src.unet_builder import UNetBuilder

schema = json.load(open("outputs/code_ready_unet/unet_ronneberger_2015.json"))

model = UNetBuilder(schema)

# Use a standard UNet input size
x = torch.randn(
    1,
    schema["input"]["channels"] or 1,
    256,
    256
)

y = model(x)

print("Output shape:", y.shape)
