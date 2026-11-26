#!/usr/bin/env python3
"""Experiment runner — one pruning experiment per call.

This script is a thin entry point that calls legacy_research.prune.prune() and
ensures results are saved in a consistent format suitable for aggregation.

Usage::

    python scripts/run_baseline_pruning.py \
        --arch resnet56 \
        --dataset cifar10 \
        --criterion l1 \
        --target_ratio 0.5

    # With fine-tuning settings:
    python scripts/run_baseline_pruning.py \
        --arch vgg16 \
        --dataset cifar100 \
        --criterion taylor \
        --target_ratio 0.7 \
        --step_ratio 0.05 \
        --finetune_epochs 2

Results are saved under legacy_research/results/ with unique timestamped names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support invocation from any current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    # Re-use the legacy pruning CLI without importing the main application.
    import legacy_research.prune as prune_module

    args = prune_module.parse_args()
    result = prune_module.prune(args)

    # Print a one-line summary suitable for shell scripting
    print(
        f"DONE | arch={args.arch} | dataset={args.dataset} | "
        f"criterion={args.criterion} | ratio={args.target_ratio:.0%} | "
        f"test_acc={result['final_test_acc']:.2f}% | "
        f"acc_drop={result['acc_drop']:.2f}% | "
        f"params_reduction={result['params_reduction_pct']:.1f}% | "
        f"flops_reduction={result['flops_reduction_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
