"""CIFAR-10 and CIFAR-100 dataset builders with deterministic train/val splits.

Normalization statistics:
  CIFAR-10:  mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
  CIFAR-100: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)

Split sizes (CIFAR-10/100 both have 50k train + 10k test):
  Train: 45,000 | Val: 5,000 | Test: 10,000
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Per-dataset normalization constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _get_cifar_transforms(mean: tuple, std: tuple):
    """Return (train_transform, eval_transform) for a CIFAR dataset."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, eval_transform


def _deterministic_split(dataset_size: int, val_size: int, seed: int):
    """Return (train_indices, val_indices) from a deterministic shuffle.

    Args:
        dataset_size: Total number of samples in the base split.
        val_size: Number of samples to allocate to validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices) as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def _load_cifar(dataset_cls, root: str, train: bool, transform, download: bool):
    try:
        return dataset_cls(root=root, train=train, download=download, transform=transform)
    except RuntimeError as exc:
        if not download:
            dataset_name = getattr(dataset_cls, "__name__", type(dataset_cls).__name__)
            raise FileNotFoundError(
                f"{dataset_name} is missing or incomplete under {root}. "
                "Place the dataset in that directory or explicitly rerun with --download."
            ) from exc
        raise


def build_cifar10_datasets(
    root: str = "./data",
    val_size: int = 5000,
    seed: int = 42,
    download: bool = False,
):
    """Build CIFAR-10 train / val / test datasets.

    Args:
        root: Directory for dataset download/cache.
        val_size: Number of training samples reserved for validation.
        seed: Seed for the deterministic train/val split.

    Returns:
        dict with keys 'train', 'val', 'test', 'num_classes', 'input_shape'.
    """
    train_tf, eval_tf = _get_cifar_transforms(CIFAR10_MEAN, CIFAR10_STD)

    # Download full train and test sets
    full_train = _load_cifar(datasets.CIFAR10, root, True, train_tf, download)
    full_train_eval = _load_cifar(datasets.CIFAR10, root, True, eval_tf, download)
    test_set = _load_cifar(datasets.CIFAR10, root, False, eval_tf, download)

    train_idx, val_idx = _deterministic_split(len(full_train), val_size, seed)

    train_set = Subset(full_train, train_idx)
    # Validation uses eval transform (no augmentation)
    val_set = Subset(full_train_eval, val_idx)

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "num_classes": 10,
        "input_shape": (3, 32, 32),
    }


def build_cifar100_datasets(
    root: str = "./data",
    val_size: int = 5000,
    seed: int = 42,
    download: bool = False,
):
    """Build CIFAR-100 train / val / test datasets.

    Args:
        root: Directory for dataset download/cache.
        val_size: Number of training samples reserved for validation.
        seed: Seed for the deterministic train/val split.

    Returns:
        dict with keys 'train', 'val', 'test', 'num_classes', 'input_shape'.
    """
    train_tf, eval_tf = _get_cifar_transforms(CIFAR100_MEAN, CIFAR100_STD)

    full_train = _load_cifar(datasets.CIFAR100, root, True, train_tf, download)
    full_train_eval = _load_cifar(datasets.CIFAR100, root, True, eval_tf, download)
    test_set = _load_cifar(datasets.CIFAR100, root, False, eval_tf, download)

    train_idx, val_idx = _deterministic_split(len(full_train), val_size, seed)

    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "num_classes": 100,
        "input_shape": (3, 32, 32),
    }
