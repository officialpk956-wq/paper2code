import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard YOLOv3 Convolutional Block (Conv -> BatchNorm -> LeakyReLU)."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """YOLOv3 Darknet-53 Residual Block."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, 1, 1, 0), ConvBlock(channels // 2, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class YOLOv3Builder(nn.Module):
    """
    YOLOv3 Architecture (Darknet-53 Backbone + Detection Heads).
    This is an executable PyTorch blueprint.
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes

        # Darknet-53 Backbone (simplified)
        self.initial = ConvBlock(3, 32, 3, 1, 1)

        self.stage1 = self._make_stage(32, 64, 1)
        self.stage2 = self._make_stage(64, 128, 2)
        self.stage3 = self._make_stage(128, 256, 8)  # Output 1 for FPN
        self.stage4 = self._make_stage(256, 512, 8)  # Output 2 for FPN
        self.stage5 = self._make_stage(512, 1024, 4)  # Output 3 for FPN

        # Detection Heads (simplified multi-scale)
        self.head_large = nn.Conv2d(1024, 3 * (5 + num_classes), 1)
        self.head_medium = nn.Conv2d(512, 3 * (5 + num_classes), 1)
        self.head_small = nn.Conv2d(256, 3 * (5 + num_classes), 1)

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = [ConvBlock(in_channels, out_channels, 3, 2, 1)]  # Downsample
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)

        # Backbone feature extraction
        x = self.stage1(x)
        x = self.stage2(x)
        out_small = self.stage3(x)  # High res
        out_medium = self.stage4(out_small)  # Mid res
        out_large = self.stage5(out_medium)  # Low res

        # Detection predictions at 3 scales
        pred_large = self.head_large(out_large)
        pred_medium = self.head_medium(out_medium)
        pred_small = self.head_small(out_small)

        return [pred_large, pred_medium, pred_small]
