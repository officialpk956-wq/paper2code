import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def forward(self, x):
        return F.hardswish(x)


class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.hardsigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, squeeze_ch):
        super().__init__()
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale)
        return x * scale


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=False, use_hs=False):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch

        hidden_dim = in_ch * expand_ratio
        self.expand = expand_ratio != 1

        layers = []
        if self.expand:
            layers.extend(
                [
                    nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    HardSwish() if use_hs else nn.ReLU(inplace=True),
                ]
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
                HardSwish() if use_hs else nn.ReLU(inplace=True),
            ]
        )

        self.conv = nn.Sequential(*layers)

        self.use_se = use_se
        if use_se:
            # typical squeeze ratio in mobilenet v3 is 1/4 of expanded ch
            squeeze_ch = max(1, hidden_dim // 4)
            self.se = SqueezeExcitation(hidden_dim, squeeze_ch)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            out = self.se(out)
        out = self.project(out)

        if self.use_res_connect:
            return x + out
        return out
