import json
import torch
from src.transformer_builder import TransformerBuilder

schema = json.load(open("outputs/code_ready/attention_all_you_need_2017.json"))

model = TransformerBuilder(schema)

x = torch.randint(0, 10000, (2, 32))
y = model(x)

print("Output shape:", y.shape)
