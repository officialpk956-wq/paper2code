import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        # gap
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.silu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, stride, se_ratio=0.25):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        hidden_dim = in_ch * expand_ratio
        if expand_ratio != 1:
            layers.extend(
                [nn.Conv2d(in_ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU()]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            ]
        )

        self.conv = nn.Sequential(*layers)
        self.se = SqueezeExcitation(hidden_dim, reduction=int(1 / se_ratio))
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out


class EfficientNetStage(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, stride, num_blocks):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MBConvBlock(
                    in_ch if i == 0 else out_ch, out_ch, expand_ratio, stride if i == 0 else 1
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
