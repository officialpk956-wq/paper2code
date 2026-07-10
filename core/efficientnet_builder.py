import math

import torch
import torch.nn as nn

from core.blocks_efficientnet import EfficientNetStage


def round_to_8(val):
    new_val = max(8, int(val + 8 / 2) // 8 * 8)
    if new_val < 0.9 * val:
        new_val += 8
    return new_val


class EfficientNetBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        w_coeff = schema["width_coeff"]
        d_coeff = schema["depth_coeff"]

        in_ch = round_to_8(32 * w_coeff)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
        )

        stages = []
        for stage_cfg in schema["stages"]:
            out_ch = round_to_8(stage_cfg["out_ch"] * w_coeff)
            num_blocks = int(math.ceil(stage_cfg["num_blocks"] * d_coeff))

            stages.append(
                EfficientNetStage(
                    in_ch, out_ch, stage_cfg["expand_ratio"], stage_cfg["stride"], num_blocks
                )
            )
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        last_ch = round_to_8(1280 * w_coeff)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(last_ch, schema["num_classes"])

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
