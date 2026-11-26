"""Training script for CIFAR-scale CNN baselines.

Trains any supported architecture on any supported dataset from scratch to
convergence. One CLI call = one (arch, dataset) training run.

Training recipe (follows standard CIFAR literature):
  - Optimizer: SGD, momentum=0.9, weight_decay=1e-4
  - LR schedule: Cosine annealing (Loshchilov & Hutter, 2017)
  - Epochs: 200 (configurable)
  - Batch size: 128

Reference:
  Loshchilov, I., & Hutter, F. (2017).
  SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR.

Usage::

    python -m src.train --arch resnet56 --dataset cifar10
    python -m src.train --arch vgg16 --dataset cifar100 --epochs 200
    python -m src.train --arch mobilenetv2 --dataset cifar10 --width_mult 0.75
    python -m src.train --arch resnet56 --dataset cifar10 --epochs 1 --smoke_test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from legacy_research.data.factory import get_dataloaders
from legacy_research.models import build_model
from legacy_research.pruning.utils import count_parameters, count_flops, model_size_mb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, topk: tuple = (1,)) -> dict:
    """Evaluate accuracy. Returns dict with 'loss', 'acc@1' (and 'acc@5' if requested)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = {k: 0 for k in topk}
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

        maxk = max(topk)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_mask = pred.eq(targets.view(1, -1).expand_as(pred))
        for k in topk:
            correct[k] += correct_mask[:k].reshape(-1).sum().item()

    result = {"loss": total_loss / total if total > 0 else float("nan")}
    for k in topk:
        result[f"acc@{k}"] = 100.0 * correct[k] / total if total > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: nn.Module,
    scheduler,
    device,
    scaler=None,
) -> dict:
    """One training epoch. Returns dict with 'loss', 'acc@1'."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    return {
        "loss": total_loss / total if total > 0 else float("nan"),
        "acc@1": 100.0 * correct / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # ---- Data ----
    logger.info("Loading dataset: %s", args.dataset)
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        download=args.download,
    )
    num_classes = meta["num_classes"]
    input_shape = meta["input_shape"]
    logger.info("Classes: %d | Input shape: %s", num_classes, input_shape)

    # ---- Model ----
    logger.info("Building model: %s", args.arch)
    model = build_model(args.arch, num_classes=num_classes, width_multiplier=args.width_mult)
    model = model.to(device)

    n_params = count_parameters(model)
    n_flops = count_flops(model, input_shape, device)
    size_mb = model_size_mb(model)
    logger.info("Params: %s | FLOPs: %s | Size: %.2f MB", f"{n_params:,}", f"{n_flops:,}", size_mb)

    # ---- Optimizer & Scheduler ----
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    epochs = 1 if args.smoke_test else args.epochs
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss()

    # AMP scaler for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and args.amp else None

    # ---- Output paths ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / f"{args.arch}_{args.dataset}_baseline.pt"
    log_csv = results_dir / f"{args.arch}_{args.dataset}_train_log.csv"
    result_json = results_dir / f"{args.arch}_{args.dataset}_baseline_result.json"
    existing_outputs = [p for p in (best_ckpt, log_csv, result_json) if p.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite prior training outputs: {paths}. "
            "Choose fresh checkpoint_dir and results_dir values."
        )

    # ---- Top-k setup ----
    topk = (1, 5) if num_classes >= 5 else (1,)

    # ---- Training loop ----
    best_val_acc = 0.0
    best_epoch = 0
    csv_rows: list[dict] = []

    logger.info("Starting training for %d epoch(s)...", epochs)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device, scaler)
        val_metrics = evaluate(model, val_loader, device, topk=topk)

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc1": round(train_metrics["acc@1"], 3),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc1": round(val_metrics["acc@1"], 3),
        }
        if "acc@5" in val_metrics:
            row["val_acc5"] = round(val_metrics["acc@5"], 3)

        csv_rows.append(row)

        # Save best checkpoint
        if val_metrics["acc@1"] > best_val_acc:
            best_val_acc = val_metrics["acc@1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "input_shape": input_shape,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "width_mult": args.width_mult,
                },
                best_ckpt,
            )

        if epoch % args.log_interval == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | train_acc=%.2f%% | "
                "val_loss=%.4f | val_acc=%.2f%% | best=%.2f%% (ep %d) | %.1fs elapsed",
                epoch, epochs,
                train_metrics["loss"], train_metrics["acc@1"],
                val_metrics["loss"], val_metrics["acc@1"],
                best_val_acc, best_epoch,
                elapsed,
            )

    # ---- Write CSV log ----
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(log_csv, "x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Training log saved to %s", log_csv)

    # ---- Final test evaluation ----
    logger.info("Loading best checkpoint (epoch %d, val_acc=%.2f%%) for test evaluation...",
                best_epoch, best_val_acc)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device, topk=topk)

    test_result = {
        "arch": args.arch,
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 3),
        "test_acc1": round(test_metrics["acc@1"], 3),
        "params": n_params,
        "flops": n_flops,
        "model_size_mb": size_mb,
    }
    if "acc@5" in test_metrics:
        test_result["test_acc5"] = round(test_metrics["acc@5"], 3)

    with open(result_json, "x") as f:
        json.dump(test_result, f, indent=2)

    logger.info("=" * 60)
    logger.info("FINAL TEST ACCURACY: %.2f%%", test_metrics["acc@1"])
    if "acc@5" in test_metrics:
        logger.info("FINAL TEST ACC@5: %.2f%%", test_metrics["acc@5"])
    logger.info("Result saved to %s", result_json)
    logger.info("=" * 60)

    return test_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-variant CNN baseline from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core
    parser.add_argument("--arch", type=str, required=True,
                        choices=["resnet56", "vgg16", "mobilenetv2"],
                        help="Model architecture.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "custom"],
                        help="Dataset name.")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"),
                        help="Root directory for dataset storage/download.")
    parser.add_argument("--download", action="store_true",
                        help="Allow downloading CIFAR datasets when absent.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial SGD learning rate.")
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="Minimum LR for cosine annealing (eta_min).")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="SGD weight decay (L2 regularization).")

    # Model extras
    parser.add_argument("--width_mult", type=float, default=1.0,
                        help="MobileNetV2 width multiplier.")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image size (for custom dataset resizing).")

    # Infrastructure
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed.")
    parser.add_argument("--device", type=str, default="",
                        help="Device override ('cuda', 'cpu'). Auto-detected if empty.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (CUDA only).")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print training metrics every N epochs.")
    output_root = Path(__file__).resolve().parent / "results"
    parser.add_argument("--checkpoint_dir", type=str, default=str(output_root / "checkpoints"),
                        help="Directory to save best checkpoints.")
    parser.add_argument("--results_dir", type=str, default=str(output_root),
                        help="Directory for CSV/JSON logs.")

    # Smoke test
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run only 1 epoch for a quick sanity check.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
