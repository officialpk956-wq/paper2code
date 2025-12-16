# src/model_builder.py

import torch.nn as nn
from src.blocks_resnet import Bottleneck


class ResNetBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()

        self.in_channels = schema["stem"]["params"]["out_channels"]

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stages = nn.ModuleList()
        for stage in schema["stages"]:
            self.stages.append(self._make_stage(stage))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            self.in_channels,
            schema["output"]["num_classes"] or 1000
        )

    def _make_stage(self, stage):
        blocks = []

        blocks.append(
            Bottleneck(
                in_channels=stage["in_channels"],
                out_channels=stage["out_channels"],
                stride=stage["stride"],
                downsample=stage["downsample"]
            )
        )

        self.in_channels = stage["out_channels"] * stage["expansion"]

        for _ in range(1, stage["num_blocks"]):
            blocks.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    out_channels=stage["out_channels"]
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)
