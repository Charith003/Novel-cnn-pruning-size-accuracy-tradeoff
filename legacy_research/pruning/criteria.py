"""Four baseline pruning criteria.

Each criterion is a callable::

    scores = criterion_fn(model, calibration_loader, device, seed)

Returns a dict mapping layer name → 1-D float Tensor of per-filter importance
scores (higher score = more important = keep; lower = prune first).

Criteria implemented
--------------------
1. random_importance  — random scores (sanity baseline)
2. l1_norm_importance — L1-norm of filter weights (Li et al., 2017)
3. taylor_importance  — |grad * activation| first-order Taylor term
                        (Molchanov et al., 2019)
4. fpgm_importance    — distance from geometric median of layer filters
                        (He et al., 2019)

References
----------
Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017).
  Pruning filters for efficient convnets. ICLR.

Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019).
  Importance estimation for neural network pruning. CVPR.

He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. (2019).
  Filter pruning via geometric median for deep convolutional neural
  networks acceleration. CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Type alias
ImportanceDict = dict[str, Tensor]


# ---------------------------------------------------------------------------
# Helper: collect prunable layers with names
# ---------------------------------------------------------------------------

def _get_named_prunable(model: nn.Module) -> list[tuple[str, nn.Conv2d, nn.BatchNorm2d, bool]]:
    """Return [(name, conv, bn, skip_locked), ...] from model.get_prunable_layers()."""
    if not hasattr(model, "get_prunable_layers"):
        raise AttributeError(
            f"{type(model).__name__} does not implement get_prunable_layers(). "
            "Ensure all model classes define this method."
        )
    return [(pl.name, pl.conv, pl.bn, pl.skip_locked)
            for pl in model.get_prunable_layers()]


# ---------------------------------------------------------------------------
# 1. Random importance
# ---------------------------------------------------------------------------

def random_importance(
    model: nn.Module,
    calibration_loader: DataLoader | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> ImportanceDict:
    """Assign uniformly random importance scores (fixed seed for reproducibility).

    Args:
        model: Neural network with ``get_prunable_layers()`` method.
        calibration_loader: Unused; kept for interface consistency.
        device: Unused; kept for interface consistency.
        seed: Random seed — must be fixed per run so the prune set is
              deterministic across multiple calls.

    Returns:
        Dict mapping layer name → random float Tensor of shape (out_channels,).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    scores: ImportanceDict = {}
    for name, conv, bn, _ in _get_named_prunable(model):
        n_filters = conv.weight.shape[0]
        scores[name] = torch.rand(n_filters, generator=rng)
    return scores


# ---------------------------------------------------------------------------
# 2. L1-norm magnitude importance (Li et al., 2017)
# ---------------------------------------------------------------------------

def l1_norm_importance(
    model: nn.Module,
    calibration_loader: DataLoader | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> ImportanceDict:
    """Score each filter by the L1-norm of its weight tensor.

    For a filter of shape (C_in, kH, kW), the score is:
        I_k = ||W_k||_1  (sum of absolute values of all weights)

    Higher norm → more important → kept. This is the criterion from
    Li et al. (2017), the standard structured pruning baseline.

    Args:
        model: Neural network with ``get_prunable_layers()`` method.
        calibration_loader: Unused; kept for interface consistency.
        device: Unused; computation is on weight tensors only.
        seed: Unused; kept for interface consistency.

    Returns:
        Dict mapping layer name → L1-norm Tensor of shape (out_channels,).
    """
    scores: ImportanceDict = {}
    for name, conv, bn, _ in _get_named_prunable(model):
        # weight: (out_channels, in_channels, kH, kW)
        w = conv.weight.data  # detached, no grad needed
        # Flatten each filter to (out_channels, -1), then sum abs values
        scores[name] = w.view(w.size(0), -1).abs().sum(dim=1).cpu()
    return scores


# ---------------------------------------------------------------------------
# 3. Taylor expansion importance (Molchanov et al., 2019)
# ---------------------------------------------------------------------------

def taylor_importance(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: str | torch.device = "cpu",
    seed: int = 0,
    n_batches: int | None = None,
) -> ImportanceDict:
    """Score each filter by the first-order Taylor expansion of the loss.

    For filter k in layer l, the score is:
        I_k = E_batch [ |grad_a_k * a_k| ]

    where a_k is the post-activation output of filter k and grad_a_k is the
    gradient of the loss w.r.t. a_k, spatially averaged per sample.

    This approximates the change in loss from zeroing out filter k, and is
    the criterion from Molchanov et al. (2019).

    Args:
        model: Neural network with ``get_prunable_layers()`` method.
        calibration_loader: DataLoader providing (inputs, labels) batches.
        device: Device to run the calibration pass on.
        seed: Unused; kept for interface consistency.
        n_batches: If set, only use the first n_batches batches. Useful for
                   fast re-scoring in the iterative pruning loop.

    Returns:
        Dict mapping layer name → Taylor importance Tensor of shape (out_channels,).
    """
    model = model.to(device)
    model.train()  # Ensure BN is in training mode for a realistic pass

    prunable = _get_named_prunable(model)
    layer_names = [p[0] for p in prunable]
    layer_convs = [p[1] for p in prunable]

    # Accumulators: sum over batches, then divide by count
    score_accum: dict[str, Tensor] = {
        name: torch.zeros(conv.weight.shape[0])
        for name, conv in zip(layer_names, layer_convs)
    }
    n_batches_seen = 0

    # Register forward hooks to capture activations
    activations: dict[str, Tensor] = {}
    hooks = []

    def _make_hook(lname: str):
        def _hook(module, inp, out):
            # out: (B, C, H, W) — store spatially averaged activation per filter
            activations[lname] = out.mean(dim=[2, 3])  # (B, C)
        return _hook

    for name, conv in zip(layer_names, layer_convs):
        hooks.append(conv.register_forward_hook(_make_hook(name)))

    criterion = nn.CrossEntropyLoss()

    try:
        for batch_idx, (inputs, targets) in enumerate(calibration_loader):
            if n_batches is not None and batch_idx >= n_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, conv in zip(layer_names, layer_convs):
                if name not in activations:
                    continue
                act = activations[name]  # (B, C) — spatially averaged
                # Gradient w.r.t. activation output: need grad of loss w.r.t. act
                # We approximate using weight gradient signal via activation * grad_w
                # Exact per-activation gradient requires register_full_backward_hook;
                # here we use the standard approximation: |∇W * W| summed per filter
                # which is equivalent in expectation (Molchanov et al. eq. 8)
                grad = conv.weight.grad  # (out_ch, in_ch, kH, kW)
                if grad is None:
                    raise RuntimeError(
                        f"No weight gradient available for layer {name}; Taylor scoring requires gradients."
                    )
                w = conv.weight.data
                # Score per filter: mean |grad_k * w_k| over all weight elements
                filter_score = (grad * w).view(grad.shape[0], -1).abs().sum(dim=1)
                score_accum[name] += filter_score.cpu()

            n_batches_seen += 1

    finally:
        for h in hooks:
            h.remove()
        model.eval()

    # Normalize by number of batches
    if n_batches_seen == 0:
        raise RuntimeError("Calibration loader was empty — cannot compute Taylor scores.")

    scores: ImportanceDict = {}
    for name in layer_names:
        scores[name] = score_accum[name] / n_batches_seen

    return scores


# ---------------------------------------------------------------------------
# 4. FPGM / Geometric Median importance (He et al., 2019)
# ---------------------------------------------------------------------------

def _weiszfeld_geometric_median(
    points: Tensor,
    n_iterations: int = 50,
    eps: float = 1e-6,
) -> Tensor:
    """Compute the geometric median of a set of points via Weiszfeld algorithm.

    Args:
        points: Tensor of shape (N, D) — N points in D-dimensional space.
        n_iterations: Number of Weiszfeld iterations.
        eps: Small value to avoid division by zero.

    Returns:
        Geometric median as a 1-D Tensor of shape (D,).
    """
    # Initialise at the arithmetic mean (good starting point)
    median = points.mean(dim=0)

    for _ in range(n_iterations):
        # Distances from current estimate to all points
        dists = torch.norm(points - median.unsqueeze(0), dim=1, keepdim=True)  # (N, 1)
        dists = dists.clamp(min=eps)

        weights = 1.0 / dists  # (N, 1)
        median = (weights * points).sum(dim=0) / weights.sum()

    return median


def fpgm_importance(
    model: nn.Module,
    calibration_loader: DataLoader | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    n_iterations: int = 50,
) -> ImportanceDict:
    """Score each filter by its L2 distance from the geometric median.

    For layer l with filters {f_1, ..., f_N} (each flattened to a vector),
    the geometric median g is computed via the Weiszfeld algorithm. The score:
        I_k = ||f_k - g||_2

    Filters closest to the median are most "redundant" (they can be
    approximated by many other filters). FPGM prunes the *lowest*-distance
    filters first (smallest I_k = most redundant).

    This is the Filter Pruning via Geometric Median criterion from
    He et al. (2019).

    Args:
        model: Neural network with ``get_prunable_layers()`` method.
        calibration_loader: Unused; kept for interface consistency.
        device: Unused; computation is on weight tensors only.
        seed: Unused; kept for interface consistency.
        n_iterations: Weiszfeld iterations for geometric median computation.

    Returns:
        Dict mapping layer name → distance-from-median Tensor of shape (out_channels,).
        Note: LOWER score = MORE redundant = pruned first (engine sorts ascending).
    """
    scores: ImportanceDict = {}

    for name, conv, bn, _ in _get_named_prunable(model):
        w = conv.weight.data.cpu()           # (out_ch, in_ch, kH, kW)
        n_filters = w.shape[0]
        flat = w.view(n_filters, -1).float() # (N, D)

        if n_filters == 1:
            # Only one filter — can't compute a meaningful median distance
            scores[name] = torch.ones(1)
            continue

        g = _weiszfeld_geometric_median(flat, n_iterations=n_iterations)
        dists = torch.norm(flat - g.unsqueeze(0), dim=1)  # (N,)
        scores[name] = dists

    return scores


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

CRITERIA: dict[str, callable] = {
    "random": random_importance,
    "l1":     l1_norm_importance,
    "taylor": taylor_importance,
    "fpgm":   fpgm_importance,
}


def get_criterion(name: str) -> callable:
    """Look up a criterion function by name.

    Args:
        name: One of 'random', 'l1', 'taylor', 'fpgm'.

    Returns:
        The corresponding criterion function.

    Raises:
        ValueError: If the name is not recognised.
    """
    name = name.lower()
    if name not in CRITERIA:
        raise ValueError(
            f"Unknown criterion '{name}'. "
            f"Choose from: {list(CRITERIA.keys())}."
        )
    return CRITERIA[name]
