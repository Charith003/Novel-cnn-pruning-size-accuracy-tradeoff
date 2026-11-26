"""Generic custom image dataset using ImageFolder conventions.

Expects the dataset root to follow the ImageFolder structure:

    <root>/
      class_A/
        img1.jpg
        img2.png
        ...
      class_B/
        ...

The dataset is deterministically split into train / val / test subsets
using the same logic as the CIFAR loaders, so the interface is identical.

Usage::

    from src.data.custom import build_custom_datasets
    splits = build_custom_datasets(
        root="/path/to/my_dataset",
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        image_size=32,
    )
    train_ds = splits["train"]
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


def _compute_dataset_stats(dataset) -> tuple[tuple, tuple]:
    """Compute per-channel mean and std over the entire dataset.

    This is relatively slow for large datasets; results can be cached by
    passing pre-computed values directly to ``build_custom_datasets``.
    """
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    for imgs, _ in loader:
        b = imgs.size(0)
        mean += imgs.view(b, 3, -1).mean(dim=[0, 2]) * b
        std += imgs.view(b, 3, -1).std(dim=[0, 2]) * b
        n_samples += b

    mean /= n_samples
    std /= n_samples
    return tuple(mean.tolist()), tuple(std.tolist())


def _make_transforms(image_size: int, mean: tuple, std: tuple):
    """Return (train_transform, eval_transform)."""
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_custom_datasets(
    root: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    image_size: int = 32,
    mean: tuple | None = None,
    std: tuple | None = None,
) -> dict:
    """Build train / val / test splits for an arbitrary ImageFolder dataset.

    Args:
        root: Path to the dataset root (ImageFolder structure).
        val_fraction: Fraction of total data for validation.
        test_fraction: Fraction of total data for test.
        seed: Random seed for deterministic splits.
        image_size: Images are resized to (image_size × image_size).
        mean: Per-channel mean for normalization. If None, computed from data.
        std: Per-channel std for normalization. If None, computed from data.

    Returns:
        dict with keys 'train', 'val', 'test', 'num_classes', 'input_shape'.
    """
    # Load without normalisation first to compute stats if needed
    raw_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    raw_dataset = datasets.ImageFolder(root=root, transform=raw_tf)

    if mean is None or std is None:
        print("Computing dataset statistics (this may take a moment) ...")
        mean, std = _compute_dataset_stats(raw_dataset)
        print(f"  mean={mean}, std={std}")

    train_tf, eval_tf = _make_transforms(image_size, mean, std)

    # Two views of the same data: augmented (train) and clean (eval)
    train_dataset = datasets.ImageFolder(root=root, transform=train_tf)
    eval_dataset = datasets.ImageFolder(root=root, transform=eval_tf)

    n_total = len(raw_dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    num_classes = len(raw_dataset.classes)

    return {
        "train": Subset(train_dataset, train_idx),
        "val": Subset(eval_dataset, val_idx),
        "test": Subset(eval_dataset, test_idx),
        "num_classes": num_classes,
        "input_shape": (3, image_size, image_size),
        "classes": raw_dataset.classes,
        "normalization": {"mean": mean, "std": std},
    }
