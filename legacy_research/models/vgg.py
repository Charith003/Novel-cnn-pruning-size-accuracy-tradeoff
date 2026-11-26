"""VGG-16 with BatchNorm for CIFAR (32×32 input).

Differences from the original Simonyan & Zisserman (2015) VGG-16:
  - BatchNorm after every Conv2d (standard practice for CIFAR training)
  - Classifier head adapted for 32×32 input: after 5 max-pool ops the
    spatial dimension collapses to 1×1, so the classifier is a simple
    512 → num_classes linear layer (no 4096-unit FC needed)
  - All conv layers are safely prunable — VGG has no skip connections

Reference:
  Simonyan, K., & Zisserman, A. (2015).
  Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
  (CIFAR adaptation is standard in the pruning literature.)

get_prunable_layers() returns ALL (conv, bn) pairs in order, all with
skip_locked=False.
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


# VGG-16 filter configuration: list of (out_channels) per conv, 'M' = MaxPool
_VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


class VGG16BN(nn.Module):
    """VGG-16 + BatchNorm adapted for CIFAR 32×32 input."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        self.features, self._conv_bn_pairs = self._make_features(_VGG16_CFG)

        # After 5 MaxPool ops on 32×32: 32 → 16 → 8 → 4 → 2 → 1
        # Spatial size = 1×1, channels = 512
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_features(self, cfg: list) -> tuple[nn.Sequential, list]:
        """Build the convolutional feature extractor and record (conv, bn) pairs."""
        layers: list[nn.Module] = []
        conv_bn_pairs: list[tuple[nn.Conv2d, nn.BatchNorm2d]] = []
        in_channels = 3

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                bn = nn.BatchNorm2d(v)
                layers.extend([conv, bn, nn.ReLU(inplace=True)])
                conv_bn_pairs.append((conv, bn))
                in_channels = v

        return nn.Sequential(*layers), conv_bn_pairs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (B, 512, 1, 1) → (B, 512)
        x = self.classifier(x)
        return x

    def get_prunable_layers(self) -> list[PrunableLayer]:
        """Return all (conv, bn) pairs. VGG has no skip connections, so all
        layers are freely prunable (skip_locked=False).

        Note: The last conv layer (conv13, 512 channels) feeds into the
        classifier linear layer. Pruning it requires also adjusting the
        classifier's in_features — the pruning engine handles this.
        """
        prunable: list[PrunableLayer] = []
        for idx, (conv, bn) in enumerate(self._conv_bn_pairs):
            prunable.append(PrunableLayer(
                name=f"features.conv{idx + 1}",
                conv=conv,
                bn=bn,
                skip_locked=False,
            ))
        return prunable


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def vgg16_bn(num_classes: int = 10) -> VGG16BN:
    """Instantiate a CIFAR VGG-16-BN model.

    Args:
        num_classes: Number of output classes.

    Returns:
        VGG16BN instance with ``get_prunable_layers()`` method.
    """
    return VGG16BN(num_classes=num_classes)
