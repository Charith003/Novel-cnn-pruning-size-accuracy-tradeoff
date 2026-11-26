"""Parameter and FLOPs counting utilities.

FLOPs are counted using ``thop`` (pip install thop). If thop is unavailable
the code falls back to a manual MACs counter.

Usage::

    from src.pruning.utils import count_parameters, count_flops, compression_stats

    params = count_parameters(model)
    flops  = count_flops(model, input_shape=(3, 32, 32))
    stats  = compression_stats(orig_params, pruned_params, orig_flops, pruned_flops)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the total number of (trainable) parameters.

    Args:
        model: Any PyTorch module.
        trainable_only: If True, count only parameters that require grad.

    Returns:
        Integer parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# FLOPs count
# ---------------------------------------------------------------------------

def count_flops(
    model: nn.Module,
    input_shape: tuple[int, int, int] = (3, 32, 32),
    device: str | torch.device = "cpu",
) -> int:
    """Count multiply-accumulate operations (MACs × 2 = FLOPs).

    Tries ``thop`` first; falls back to a manual conv/linear counter.

    Args:
        model: Neural network module.
        input_shape: (C, H, W) input dimensions (batch dimension excluded).
        device: Device to run the dummy forward pass on.

    Returns:
        Approximate total FLOPs as an integer.
    """
    model = model.to(device)
    dummy_input = torch.zeros(1, *input_shape, device=device)

    try:
        from thop import profile  # type: ignore
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return int(macs) * 2  # MACs → FLOPs

    except ImportError:
        pass  # Fall back to manual counter

    return _manual_flop_count(model, dummy_input)


def _manual_flop_count(model: nn.Module, dummy_input: torch.Tensor) -> int:
    """Manually count FLOPs for Conv2d and Linear layers via forward hooks."""
    total_flops = [0]  # list so closure can mutate it

    hooks = []

    def conv_hook(module: nn.Conv2d, inp, out):
        b, c_out, h_out, w_out = out.shape
        _, c_in, kh, kw = module.weight.shape
        groups = module.groups
        # FLOPs = 2 × (kernel_ops) × output_elements
        kernel_ops = (c_in // groups) * kh * kw
        total_flops[0] += 2 * kernel_ops * c_out * h_out * w_out * b

    def linear_hook(module: nn.Linear, inp, out):
        b = inp[0].shape[0]
        total_flops[0] += 2 * module.in_features * module.out_features * b

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    return total_flops[0]


# ---------------------------------------------------------------------------
# Compression statistics
# ---------------------------------------------------------------------------

def compression_stats(
    original_params: int,
    pruned_params: int,
    original_flops: int,
    pruned_flops: int,
) -> dict:
    """Compute compression ratios and size reduction percentages.

    Args:
        original_params: Parameter count of the unpruned model.
        pruned_params: Parameter count of the pruned model.
        original_flops: FLOPs of the unpruned model.
        pruned_flops: FLOPs of the pruned model.

    Returns:
        Dict with keys:
          - params_original, params_pruned
          - flops_original, flops_pruned
          - params_reduction_pct: % params removed
          - flops_reduction_pct: % FLOPs removed
          - compression_ratio: original / pruned parameter count
    """
    params_reduction = 100.0 * (1.0 - pruned_params / original_params) if original_params > 0 else 0.0
    flops_reduction = 100.0 * (1.0 - pruned_flops / original_flops) if original_flops > 0 else 0.0
    compression_ratio = original_params / pruned_params if pruned_params > 0 else float("inf")

    return {
        "params_original": original_params,
        "params_pruned": pruned_params,
        "flops_original": original_flops,
        "flops_pruned": pruned_flops,
        "params_reduction_pct": round(params_reduction, 2),
        "flops_reduction_pct": round(flops_reduction, 2),
        "compression_ratio": round(compression_ratio, 4),
    }


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes (parameters only, float32 assumed)."""
    n_params = count_parameters(model, trainable_only=False)
    return round(n_params * 4 / 1024 / 1024, 3)  # 4 bytes per float32
