import torch.nn as nn

from core.blocks_mobilenet import HardSwish, InvertedResidual


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        multiplier = schema["multiplier"]
        num_classes = schema["num_classes"]

        in_ch = _make_divisible(16 * multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            HardSwish(),
        )

        features = []
        for stage in schema["stages"]:
            out_ch = _make_divisible(stage["out_ch"] * multiplier)
            for i in range(stage["n"]):
                stride = stage["stride"] if i == 0 else 1
                features.append(
                    InvertedResidual(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        stride=stride,
                        expand_ratio=stage["expand"],
                        use_se=stage["se"],
                        use_hs=stage["hs"],
                    )
                )
                in_ch = out_ch

        self.features = nn.Sequential(*features)

        last_conv_ch = _make_divisible(in_ch * 6)
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, last_conv_ch, 1, bias=False), nn.BatchNorm2d(last_conv_ch), HardSwish()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        last_ch = _make_divisible(1280 * multiplier) if multiplier > 1.0 else 1280
        self.head_classifier = nn.Sequential(
            nn.Conv2d(last_conv_ch, last_ch, 1),
            HardSwish(),
            nn.Flatten(1),
            nn.Linear(last_ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = self.head_classifier(x)
        return x
