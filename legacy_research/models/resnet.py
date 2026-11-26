"""ResNet-56 for CIFAR (He et al., 2016).

Architecture spec (CIFAR version):
  - Initial conv: 3×3, 16 channels, stride=1, padding=1
  - 3 stages of 9 BasicBlocks each: channels [16, 32, 64]
  - Downsampling between stages via stride-2 convolution in shortcut
  - Global average pooling → fully connected classifier
  - Total depth: 1 + 3×(9×2) + 1 = 56 layers (hence ResNet-56)

Reference:
  He, K., Zhang, X., Ren, S., & Sun, J. (2016).
  Deep Residual Learning for Image Recognition. CVPR.
  Appendix: CIFAR-10 and CIFAR-100 experiments.

Skip-connection pruning notes:
  Layers that feed directly into a residual addition are "skip-locked":
  they cannot have their output channel count changed without also
  changing the other branch. The pruning engine uses ``get_prunable_layers()``
  which marks these appropriately via the ``skip_locked`` flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class PrunableLayer(NamedTuple):
    """A prunable (conv, bn) pair with metadata."""
    name: str          # Human-readable identifier, e.g. "layer1.0.conv1"
    conv: nn.Conv2d
    bn: nn.BatchNorm2d
    skip_locked: bool  # True if this conv feeds a residual addition directly


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two-layer residual block for CIFAR ResNet."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: either identity or 1×1 projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------------
# ResNet-56
# ---------------------------------------------------------------------------

class ResNet56(nn.Module):
    """CIFAR ResNet-56: 3 stages × 9 BasicBlocks, channels [16, 32, 64]."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Stage 0: initial 3×3 conv (stride 1 for CIFAR 32×32 input)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stages 1–3
        self.layer1 = self._make_layer(16, 16, n_blocks=9, stride=1)
        self.layer2 = self._make_layer(16, 32, n_blocks=9, stride=2)
        self.layer3 = self._make_layer(32, 64, n_blocks=9, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_prunable_layers(self) -> list[PrunableLayer]:
        """Return ordered (conv, bn) pairs available for structured pruning.

        Policy:
          - conv2 in each BasicBlock (the second conv, feeding into the
            residual addition) is marked skip_locked=True because its
            output must match the shortcut channel count.
          - conv1 in each BasicBlock (before relu, not feeding the addition
            directly) is skip_locked=False and freely prunable.
          - The initial stem conv1 is NOT included — it's a 3-channel input
            conv and pruning it is rarely useful.

        The pruning engine respects skip_locked to avoid channel mismatches.
        """
        prunable: list[PrunableLayer] = []

        for stage_name in ("layer1", "layer2", "layer3"):
            stage: nn.Sequential = getattr(self, stage_name)
            for block_idx, block in enumerate(stage):
                block: BasicBlock
                # conv1 of each block: freely prunable
                prunable.append(PrunableLayer(
                    name=f"{stage_name}.{block_idx}.conv1",
                    conv=block.conv1,
                    bn=block.bn1,
                    skip_locked=False,
                ))
                # conv2 feeds the residual add: output must match shortcut
                prunable.append(PrunableLayer(
                    name=f"{stage_name}.{block_idx}.conv2",
                    conv=block.conv2,
                    bn=block.bn2,
                    skip_locked=True,
                ))

        return prunable


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def resnet56(num_classes: int = 10) -> ResNet56:
    """Instantiate a CIFAR ResNet-56 model.

    Args:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100).

    Returns:
        ResNet56 instance with ``get_prunable_layers()`` method.
    """
    return ResNet56(num_classes=num_classes)
