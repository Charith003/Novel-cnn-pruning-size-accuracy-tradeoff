"""MobileNetV2 for CIFAR (32×32 input).

Differences from the standard ImageNet MobileNetV2 (Sandler et al., 2018):
  - First conv stride=1 (not 2): prevents over-downsampling 32×32 inputs
  - No initial stride-2 in the first inverted residual block either
  - Width multiplier (width_mult) is fully configurable for ablations

Reference:
  Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
  MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.

Pruning notes:
  MobileNetV2 uses depthwise separable convolutions.
  For structured pruning we expose only the *pointwise* (1×1) conv layers
  as prunable, since:
    - Depthwise convs have only 1 filter per channel group — pruning them
      means pruning the entire channel, which is handled by pruning the
      preceding pointwise conv.
    - The pointwise convs control the actual channel counts and are where
      the compression comes from.

  skip_locked is set True for the final pointwise conv in each inverted
  residual block that has an identity skip connection (i.e., when
  in_channels == out_channels and stride == 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import NamedTuple


class PrunableLayer(NamedTuple):
    name: str
    conv: nn.Conv2d
    bn: nn.BatchNorm2d
    skip_locked: bool


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU6(nn.Sequential):
    """Standard conv → BN → ReLU6 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted residual block (bottleneck) from MobileNetV2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ):
        super().__init__()
        assert stride in (1, 2)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            # Pointwise expand
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
        # Depthwise
        layers.append(ConvBNReLU6(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        # Pointwise project (no activation — "linear bottleneck")
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

        # Store references for pruning
        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Locate the expand pointwise and project pointwise for pruning
        if expand_ratio != 1:
            self._expand_conv: nn.Conv2d = layers[0][0]
            self._expand_bn: nn.BatchNorm2d = layers[0][1]
        else:
            self._expand_conv = None
            self._expand_bn = None

        # Project conv is always the last conv
        self._project_conv: nn.Conv2d = layers[-2]
        self._project_bn: nn.BatchNorm2d = layers[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 with CIFAR-adapted strides and configurable width."""

    # (expand_ratio, out_channels, n_blocks, stride)
    # stride of first block in each group; subsequent blocks use stride=1
    # For CIFAR: first stride=2 block changed to stride=1 to preserve resolution
    _INVERTED_RESIDUAL_SETTINGS = [
        (1,  16,  1, 1),
        (6,  24,  2, 1),   # ImageNet: stride=2 → CIFAR: stride=1
        (6,  32,  3, 2),
        (6,  64,  4, 2),
        (6,  96,  3, 1),
        (6, 160,  3, 2),
        (6, 320,  1, 1),
    ]

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult

        input_channel = _make_divisible(32 * width_mult, 8)
        last_channel = _make_divisible(1280 * max(1.0, width_mult), 8)

        # Initial conv — stride=1 for CIFAR (not 2 like ImageNet)
        self.first_conv = ConvBNReLU6(3, input_channel, stride=1)
        self._first_conv_pair = (self.first_conv[0], self.first_conv[1])

        # Inverted residual blocks
        self.blocks: nn.ModuleList = nn.ModuleList()
        self._block_settings: list = []  # (expand, out, stride) for reference

        for t, c, n, s in self._INVERTED_RESIDUAL_SETTINGS:
            out_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                block = InvertedResidual(input_channel, out_channel, stride, t)
                self.blocks.append(block)
                self._block_settings.append((t, out_channel, stride))
                input_channel = out_channel

        # Last conv (pointwise expansion)
        self.last_conv = ConvBNReLU6(input_channel, last_channel, kernel_size=1)
        self._last_conv_pair = (self.last_conv[0], self.last_conv[1])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_prunable_layers(self) -> list[PrunableLayer]:
        """Return pointwise (conv, bn) pairs available for structured pruning.

        Depthwise convs are NOT included — they track their input channel
        count from the preceding expand pointwise conv.

        skip_locked rules:
          - Expand pointwise: never skip_locked (output feeds depthwise, not addition)
          - Project pointwise: skip_locked if:
              (a) This block has use_res_connect (its output is ADDED to its input), OR
              (b) The NEXT block has use_res_connect (the next block's identity shortcut
                  carries THIS block's output, so this block's output must stay fixed).
        """
        prunable: list[PrunableLayer] = []
        n_blocks = len(self.blocks)

        for block_idx, block in enumerate(self.blocks):
            block: InvertedResidual

            # Next block for look-ahead
            next_block = self.blocks[block_idx + 1] if block_idx + 1 < n_blocks else None

            # (a) Expand pointwise (only exists when expand_ratio > 1)
            if block.expand_ratio != 1 and block._expand_conv is not None:
                prunable.append(PrunableLayer(
                    name=f"blocks.{block_idx}.expand_pw",
                    conv=block._expand_conv,
                    bn=block._expand_bn,
                    skip_locked=False,  # Expand conv is never on skip path
                ))

            # (b) Project pointwise — skip_locked if this OR next block uses residual
            next_uses_skip = next_block is not None and next_block.use_res_connect
            prunable.append(PrunableLayer(
                name=f"blocks.{block_idx}.project_pw",
                conv=block._project_conv,
                bn=block._project_bn,
                skip_locked=block.use_res_connect or next_uses_skip,
            ))

        return prunable


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def mobilenet_v2(num_classes: int = 10, width_mult: float = 1.0) -> MobileNetV2:
    """Instantiate a CIFAR MobileNetV2 model.

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier (0.35, 0.5, 0.75, 1.0 are common).

    Returns:
        MobileNetV2 instance with ``get_prunable_layers()`` method.
    """
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult)
