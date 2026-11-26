"""Pruning CLI — prune a single trained checkpoint.

Load a trained baseline, run iterative_prune_and_finetune, save the pruned
checkpoint, and log results. This is the thin CLI wrapper around the engine.

Usage::

    python -m legacy_research.prune \
        --arch resnet56 \
        --dataset cifar10 \
        --criterion l1 \
        --target_ratio 0.5

The checkpoint is loaded from legacy_research/results/checkpoints/{arch}_{dataset}_baseline.pt by
default, or you can specify --checkpoint explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

from legacy_research.data.factory import get_dataloaders
from legacy_research.models import build_model
from legacy_research.pruning.engine import iterative_prune_and_finetune
from legacy_research.pruning.utils import count_parameters, count_flops, compression_stats, model_size_mb
from legacy_research.train import evaluate, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prune")


def load_pruned_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load a structurally pruned model checkpoint created by this module.

    The checkpoint contains a pickled model object because a state dict alone
    cannot recreate the physically changed channel topology. Only load files
    produced by a trusted local run.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model = payload.get("model_object")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Checkpoint does not contain a legacy pruned model object.")
    return model.to(device)


def prune(args: argparse.Namespace):
    started_at = time.perf_counter()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # ---- Data ----
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

    # ---- Model ----
    model = build_model(args.arch, num_classes=num_classes, width_multiplier=args.width_mult)

    # ---- Load checkpoint ----
    default_ckpt = Path(__file__).resolve().parent / "results" / "checkpoints" / f"{args.arch}_{args.dataset}_baseline.pt"
    ckpt_path = args.checkpoint or str(default_ckpt)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Run 'python -m src.train --arch {args.arch} --dataset {args.dataset}' first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    logger.info("Loaded checkpoint: %s (val_acc=%.2f%%)", ckpt_path, ckpt.get("val_acc", 0))
    model = model.to(device)

    # ---- Baseline stats ----
    orig_params = count_parameters(model)
    orig_flops = count_flops(model, input_shape, device)
    topk = (1, 5) if num_classes >= 5 else (1,)
    base_metrics = evaluate(model, test_loader, device, topk=topk)
    logger.info(
        "Baseline | params=%s | flops=%s | test_acc=%.2f%%",
        f"{orig_params:,}", f"{orig_flops:,}", base_metrics["acc@1"]
    )

    # ---- Prune + fine-tune ----
    logger.info(
        "Starting iterative pruning: criterion=%s  target_ratio=%.0f%%  step=%.0f%%  ft_epochs=%d",
        args.criterion, args.target_ratio * 100, args.step_ratio * 100, args.finetune_epochs,
    )

    pruned_model, step_logs = iterative_prune_and_finetune(
        model=model,
        target_ratio=args.target_ratio,
        criterion_name=args.criterion,
        calibration_loader=train_loader,
        train_loader=train_loader,
        val_loader=val_loader,
        step_ratio=args.step_ratio,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        device=device,
        arch=args.arch,
        input_shape=input_shape,
        seed=args.seed,
        verbose=True,
    )

    # ---- Final test evaluation ----
    final_metrics = evaluate(pruned_model, test_loader, device, topk=topk)
    pruned_params = count_parameters(pruned_model)
    pruned_flops = count_flops(pruned_model, input_shape, device)
    stats = compression_stats(orig_params, pruned_params, orig_flops, pruned_flops)

    logger.info("=" * 60)
    logger.info("PRUNING COMPLETE")
    logger.info("  params: %s → %s  (-%.1f%%)", f"{orig_params:,}", f"{pruned_params:,}", stats["params_reduction_pct"])
    logger.info("  flops:  %s → %s  (-%.1f%%)", f"{orig_flops:,}", f"{pruned_flops:,}", stats["flops_reduction_pct"])
    logger.info("  test_acc: %.2f%% → %.2f%%", base_metrics["acc@1"], final_metrics["acc@1"])
    logger.info("  model size: %.2f MB", model_size_mb(pruned_model))
    logger.info("=" * 60)

    # ---- Save pruned checkpoint ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ratio_str = f"{int(args.target_ratio * 100):03d}"
    run_stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    pruned_ckpt = ckpt_dir / f"{args.arch}_{args.dataset}_{args.criterion}_r{ratio_str}_{run_stamp}.pt"

    with pruned_ckpt.open("xb") as ckpt_file:
        torch.save(
            {
                "arch": args.arch,
                "dataset": args.dataset,
                "criterion": args.criterion,
                "target_ratio": args.target_ratio,
                "num_classes": num_classes,
                "input_shape": input_shape,
                "state_dict": pruned_model.state_dict(),
                "model_object": pruned_model,
                "final_test_acc": final_metrics["acc@1"],
                "compression_stats": stats,
            },
            ckpt_file,
        )
    logger.info("Pruned checkpoint saved to %s", pruned_ckpt)

    # ---- Save result JSON ----
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_json = results_dir / f"pruning_{args.arch}_{args.dataset}_{args.criterion}_r{ratio_str}_{run_stamp}.json"

    result = {
        "arch": args.arch,
        "dataset": args.dataset,
        "criterion": args.criterion,
        "target_ratio": args.target_ratio,
        "step_ratio": args.step_ratio,
        "seed": args.seed,
        "finetune_epochs": args.finetune_epochs,
        "baseline_test_acc": round(base_metrics["acc@1"], 3),
        "final_test_acc": round(final_metrics["acc@1"], 3),
        "acc_drop": round(base_metrics["acc@1"] - final_metrics["acc@1"], 3),
        **{k: v for k, v in stats.items()},
        "model_size_mb": model_size_mb(pruned_model),
        "step_logs": step_logs,
        "execution_time_seconds": round(time.perf_counter() - started_at, 3),
    }
    if "acc@5" in final_metrics:
        result["final_test_acc5"] = round(final_metrics["acc@5"], 3)

    with open(result_json, "x") as f:
        json.dump(result, f, indent=2)
    logger.info("Result saved to %s", result_json)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune a trained CNN checkpoint using a selected criterion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", type=str, required=True,
                        choices=["resnet56", "vgg16", "mobilenetv2"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "custom"])
    parser.add_argument("--criterion", type=str, required=True,
                        choices=["random", "l1", "taylor", "fpgm"],
                        help="Pruning criterion.")
    parser.add_argument("--target_ratio", type=float, required=True,
                        help="Target fraction of filters to prune (e.g. 0.5 = 50%%).")
    parser.add_argument("--step_ratio", type=float, default=0.05,
                        help="Fraction to prune per iteration.")
    parser.add_argument("--finetune_epochs", type=int, default=2,
                        help="Fine-tuning epochs per pruning step.")
    parser.add_argument("--finetune_lr", type=float, default=1e-3,
                        help="Fine-tuning learning rate.")

    # Data
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--download", action="store_true", help="Allow downloading CIFAR datasets when absent.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=32)

    # Model
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to baseline checkpoint. Auto-detected if empty.")

    # Output
    output_root = Path(__file__).resolve().parent / "results"
    parser.add_argument("--checkpoint_dir", type=str, default=str(output_root / "checkpoints"))
    parser.add_argument("--results_dir", type=str, default=str(output_root))

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prune(args)
