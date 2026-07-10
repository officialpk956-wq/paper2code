import torch
import torch.nn as nn
import torch.nn.functional as F

from core.blocks_densenet import DenseBlock, TransitionLayer


class DenseNetBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        num_blocks = schema["num_blocks"]
        growth_rate = schema["growth_rate"]
        init_ch = schema["init_ch"]
        num_classes = schema["num_classes"]

        self.features = nn.Sequential(
            nn.Conv2d(3, init_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = init_ch
        for i, num_layers in enumerate(num_blocks):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(num_blocks) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
