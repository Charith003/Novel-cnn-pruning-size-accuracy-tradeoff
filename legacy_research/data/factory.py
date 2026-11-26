"""Unified dataloader factory.

All datasets are accessed through a single entry point::

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset_name="cifar10",
        data_root="./data",
        batch_size=128,
    )
    print(meta["num_classes"])   # 10
    print(meta["input_shape"])   # (3, 32, 32)

The returned ``meta`` dict always contains:
  - num_classes (int)
  - input_shape (tuple[int, int, int])  — (C, H, W)
  - dataset_name (str)
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from .cifar import build_cifar10_datasets, build_cifar100_datasets
from .custom import build_custom_datasets


def get_dataloaders(
    dataset_name: str,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
    # Custom dataset extras
    image_size: int = 32,
    custom_mean: tuple | None = None,
    custom_std: tuple | None = None,
    custom_test_fraction: float = 0.1,
    pin_memory: bool = True,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Return (train_loader, val_loader, test_loader, meta) for any dataset.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', or 'custom'.
        data_root: Root directory. For CIFAR this is the download cache;
                   for custom it must be an ImageFolder root.
        batch_size: Mini-batch size for all loaders.
        num_workers: DataLoader worker processes.
        val_fraction: Fraction of *training* data used for validation
                      (CIFAR: 5k/50k = 0.10; custom: configurable).
        seed: Global reproducibility seed.
        image_size: Only used for custom dataset resizing.
        custom_mean: Pre-computed mean for custom dataset (skip auto-compute).
        custom_std: Pre-computed std for custom dataset (skip auto-compute).
        custom_test_fraction: Test set fraction for custom dataset.
        pin_memory: Enable pin_memory for GPU training.

    Returns:
        (train_loader, val_loader, test_loader, meta)
        meta keys: num_classes, input_shape, dataset_name
    """
    name = dataset_name.lower()

    if name == "cifar10":
        val_size = int(50_000 * val_fraction)
        splits = build_cifar10_datasets(root=data_root, val_size=val_size, seed=seed, download=download)
    elif name == "cifar100":
        val_size = int(50_000 * val_fraction)
        splits = build_cifar100_datasets(root=data_root, val_size=val_size, seed=seed, download=download)
    elif name == "custom":
        splits = build_custom_datasets(
            root=data_root,
            val_fraction=val_fraction,
            test_fraction=custom_test_fraction,
            seed=seed,
            image_size=image_size,
            mean=custom_mean,
            std=custom_std,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose from: cifar10, cifar100, custom."
        )

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(
        splits["train"], shuffle=True, drop_last=True, **common_kwargs
    )
    val_loader = DataLoader(
        splits["val"], shuffle=False, **common_kwargs
    )
    test_loader = DataLoader(
        splits["test"], shuffle=False, **common_kwargs
    )

    meta = {
        "num_classes": splits["num_classes"],
        "input_shape": splits["input_shape"],
        "dataset_name": name,
    }

    return train_loader, val_loader, test_loader, meta
