"""Legacy CIFAR CNN architectures."""
from .resnet import resnet56
from .vgg import vgg16_bn
from .mobilenetv2 import mobilenet_v2


def build_model(arch: str, num_classes: int, width_multiplier: float = 1.0):
    key = arch.lower()
    if key == "resnet56":
        return resnet56(num_classes=num_classes)
    if key in ("vgg16", "vgg16_bn"):
        return vgg16_bn(num_classes=num_classes)
    if key == "mobilenetv2":
        return mobilenet_v2(num_classes=num_classes, width_mult=width_multiplier)
    raise ValueError(f"Unknown architecture {arch!r}")
