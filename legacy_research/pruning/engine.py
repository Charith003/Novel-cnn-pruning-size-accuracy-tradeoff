"""Core pruning engine.

Three public functions:

compute_importance(model, criterion_name, calibration_loader, device)
    → Dict[str, Tensor]  — per-filter importance scores for all prunable layers

prune_filters(model, importance_scores, target_ratio, arch)
    → model  — rebuilt model with lowest-scoring filters physically removed

iterative_prune_and_finetune(...)
    → (pruned_model, log_dict)  — the main experiment loop

Skip-connection safety
----------------------
Layers marked skip_locked=True by ``get_prunable_layers()`` are excluded from
individual channel-count changes. Their importance scores are computed normally
(so they appear in the ablation story), but when ``prune_filters`` selects
filters to remove, it skips any layer whose ``skip_locked`` flag is True.

This is the conservative strategy: a small number of filters are never pruned
individually. It is correct for all CIFAR ResNet configurations where the
shortcut is an identity map (no projection conv). For projection shortcuts
(stage boundaries), the shortcut itself is a Conv2d — those blocks are also
marked skip_locked because changing both branches together requires extra logic.
Users who want to prune skip-locked layers must do so by pruning entire
blocks (out of scope for Part 1).

Layer reconstruction
--------------------
After selecting the kept filter indices for a layer, the engine rebuilds:
  1. A new Conv2d with ``out_channels = len(kept_indices)``.
  2. A new BatchNorm2d of matching size.
  3. The *next* layer's Conv2d with matching ``in_channels``.

The "next layer" lookup is done via a connectivity map built from the model's
prunable layer list:

  - For most architectures (VGG, MobileNetV2 pointwise), layer i's output
    feeds layer i+1's input.
  - For ResNet, after the last conv of a block (conv2), the next consumer is
    the first conv of the next block *and* the shortcut — but conv2 is
    skip_locked so it is never modified here.

For the final prunable layer in a VGG model, the "next" consumer is the
classifier Linear layer (in_features must be updated).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from .criteria import get_criterion, ImportanceDict
from .utils import count_parameters, count_flops, compression_stats, model_size_mb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# compute_importance
# ---------------------------------------------------------------------------

def compute_importance(
    model: nn.Module,
    criterion_name: str,
    calibration_loader,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> ImportanceDict:
    """Compute per-filter importance scores for all prunable layers.

    Args:
        model: Network with ``get_prunable_layers()`` method.
        criterion_name: One of 'random', 'l1', 'taylor', 'fpgm'.
        calibration_loader: DataLoader for calibration; may be None for
                            weight-only criteria (random, l1, fpgm).
        device: Device for forward/backward passes.
        seed: Passed to the criterion for reproducibility.

    Returns:
        Dict[layer_name → importance Tensor (out_channels,)].
    """
    criterion_fn = get_criterion(criterion_name)
    model.eval()
    scores = criterion_fn(model, calibration_loader, device, seed)
    return scores


# ---------------------------------------------------------------------------
# _build_connectivity: map each prunable layer → its successor(s)
# ---------------------------------------------------------------------------

def _build_next_layer_map(model: nn.Module) -> dict[str, list[nn.Module]]:
    """For each prunable conv, find the next module whose in_channels must change.

    Returns dict mapping layer_name → list of successor nn.Module objects
    (Conv2d or Linear) whose in_channels / in_features must be updated when
    this layer's out_channels shrinks.

    The lookup is conservative and handles:
      - Sequential stacks (VGG, MobileNetV2 pointwise chains)
      - The final conv → classifier Linear transition (VGG, MobileNetV2)
    """
    prunable = model.get_prunable_layers()
    n = len(prunable)
    successors: dict[str, list[nn.Module]] = {}

    for i, pl in enumerate(prunable):
        name = pl.name
        conv = pl.conv
        successors[name] = []

        if i + 1 < n:
            next_conv = prunable[i + 1].conv
            successors[name].append(next_conv)

        # Check for terminal conv → Linear transition
        # The last conv's output may feed into a Linear classifier
        if i == n - 1 or (i + 1 < n and prunable[i + 1].skip_locked):
            # Walk model modules to find a Linear layer whose in_features
            # currently matches this conv's out_channels
            for m in model.modules():
                if isinstance(m, nn.Linear) and m.in_features == conv.weight.shape[0]:
                    if m not in successors[name]:
                        successors[name].append(m)

    return successors


# ---------------------------------------------------------------------------
# _rebuild_conv: create a new Conv2d with selected output and input channels
# ---------------------------------------------------------------------------

def _rebuild_conv(
    old_conv: nn.Conv2d,
    kept_out_indices: Tensor | None,  # None = keep all
    kept_in_indices: Tensor | None,   # None = keep all
) -> nn.Conv2d:
    """Create a new Conv2d with a subset of output and/or input channels.

    Correctly handles depthwise convolutions (groups == in_channels):
    when input channels are sliced, groups is updated to match.
    """
    w = old_conv.weight.data  # (out, in/groups, kH, kW)

    if kept_out_indices is not None:
        w = w[kept_out_indices]
    if kept_in_indices is not None:
        w = w[:, kept_in_indices]

    new_out, new_in_per_group = w.shape[0], w.shape[1]

    # Determine new number of groups
    is_depthwise = (old_conv.groups == old_conv.in_channels)
    if is_depthwise:
        # Depthwise: groups must equal new in_channels (= new_out for depthwise)
        new_in_channels = new_out  # depthwise: in_channels == out_channels
        new_groups = new_out
        # Weight for depthwise: (out, 1, kH, kW) — already correct after out slicing
    else:
        # Standard / grouped conv
        new_groups = old_conv.groups
        new_in_channels = new_in_per_group * new_groups
        if new_in_channels == 0:
            new_in_channels = 1

    new_conv = nn.Conv2d(
        new_in_channels,
        new_out,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=new_groups,
        bias=old_conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(w.clone())
    if old_conv.bias is not None:
        b = old_conv.bias.data
        if kept_out_indices is not None:
            b = b[kept_out_indices]
        new_conv.bias = nn.Parameter(b.clone())
    return new_conv


def _rebuild_bn(old_bn: nn.BatchNorm2d, kept_indices: Tensor) -> nn.BatchNorm2d:
    """Create a new BatchNorm2d keeping only the specified channel indices."""
    new_bn = nn.BatchNorm2d(
        len(kept_indices),
        eps=old_bn.eps,
        momentum=old_bn.momentum,
        affine=old_bn.affine,
        track_running_stats=old_bn.track_running_stats,
    )
    if old_bn.affine:
        new_bn.weight = nn.Parameter(old_bn.weight.data[kept_indices].clone())
        new_bn.bias = nn.Parameter(old_bn.bias.data[kept_indices].clone())
    if old_bn.track_running_stats and old_bn.running_mean is not None:
        new_bn.running_mean = old_bn.running_mean[kept_indices].clone()
        new_bn.running_var = old_bn.running_var[kept_indices].clone()
        new_bn.num_batches_tracked = old_bn.num_batches_tracked.clone()
    return new_bn


def _rebuild_linear_in(old_linear: nn.Linear, kept_in_indices: Tensor) -> nn.Linear:
    """Create a new Linear layer with a reduced in_features."""
    w = old_linear.weight.data[:, kept_in_indices]
    new_linear = nn.Linear(len(kept_in_indices), old_linear.out_features, bias=old_linear.bias is not None)
    new_linear.weight = nn.Parameter(w.clone())
    if old_linear.bias is not None:
        new_linear.bias = nn.Parameter(old_linear.bias.data.clone())
    return new_linear


# ---------------------------------------------------------------------------
# Module identity helpers
# ---------------------------------------------------------------------------

def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a submodule identified by dot-separated path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _build_id_to_name_map(model: nn.Module) -> dict[int, str]:
    """Map object id of every submodule to its dot-separated name."""
    return {id(m): name for name, m in model.named_modules()}


def _build_successor_map(
    model: nn.Module,
    prunable,
    id_to_name: dict[int, str],
) -> dict[str, list[tuple[str, nn.Module]]]:
    """For each prunable conv, find which modules consume its output channels.

    Uses module object identity to resolve the conv's actual position in
    named_modules() (which may differ from pl.name for architectures that
    store modules in Sequential containers indexed by integer).

    Returns dict: pl.name → list of (successor_module_name, successor_module)
    """
    # Build ordered list of (name, module) for all named modules
    all_named = list(model.named_modules())
    name_to_idx = {name: idx for idx, (name, _) in enumerate(all_named)}

    # Resolve each prunable conv to its real position in all_named via id
    prunable_info = []
    for pl in prunable:
        real_name = id_to_name.get(id(pl.conv), pl.name)
        idx = name_to_idx.get(real_name, -1)
        prunable_info.append((pl.name, pl.conv, idx))

    # Set of conv module ids that are prunable (for early-stop detection)
    prunable_conv_ids = {id(pl.conv) for pl in prunable}

    successors: dict[str, list[tuple[str, nn.Module]]] = {n: [] for n, _, _ in prunable_info}

    for p_name, p_conv, p_idx in prunable_info:
        if p_idx < 0:
            continue
        p_out = p_conv.weight.shape[0]  # out_channels of this prunable conv
        # Find the FIRST module after p_idx whose in_channels/in_features == p_out
        for m_name, m in all_named[p_idx + 1:]:
            if isinstance(m, nn.Conv2d) and m.in_channels == p_out:
                successors[p_name].append((m_name, m))
                break  # Only direct successor
            elif isinstance(m, nn.Linear) and m.in_features == p_out:
                successors[p_name].append((m_name, m))
                break
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                # Hit a layer with different in_channels — stop searching
                # unless it's just an activation or pooling layer.
                break

    return successors


# ---------------------------------------------------------------------------
# prune_filters
# ---------------------------------------------------------------------------

def prune_filters(
    model: nn.Module,
    importance_scores: ImportanceDict,
    target_ratio: float,
    arch: str = "generic",
) -> nn.Module:
    """Remove the lowest-scoring filters from all non-skip-locked layers.

    The function physically rebuilds Conv2d and BatchNorm2d layers in place
    on a deep copy of the model, then propagates the reduced in_channels to
    successor layers.

    Args:
        model: Network with ``get_prunable_layers()`` method.
        importance_scores: Dict from ``compute_importance``.
        target_ratio: Fraction of filters to prune globally (0 < ratio < 1).
        arch: Architecture name (for logging; not used in logic).

    Returns:
        A new model instance with filters physically removed.

    Notes:
        - skip_locked layers are skipped — their filter count is unchanged.
        - The pruning ratio is applied globally across all prunable (non-locked)
          filters, using a global score threshold. This avoids removing ALL
          filters from any single layer (which would break the model).
        - A minimum of 1 filter is always kept per layer.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be between 0 and 1 (exclusive); got {target_ratio}.")
    if not hasattr(model, "get_prunable_layers"):
        raise AttributeError(f"{type(model).__name__} does not implement get_prunable_layers().")
    for layer in model.get_prunable_layers():
        if layer.name not in importance_scores:
            continue
        scores = importance_scores[layer.name]
        if not isinstance(scores, Tensor) or scores.ndim != 1:
            raise ValueError(f"Importance scores for {layer.name} must be a 1-D Tensor.")
        if scores.numel() != layer.conv.weight.shape[0]:
            raise ValueError(
                f"Importance score length for {layer.name} ({scores.numel()}) does not match "
                f"out_channels ({layer.conv.weight.shape[0]})."
            )
        if not torch.isfinite(scores).all():
            raise ValueError(f"Importance scores for {layer.name} must be finite.")
    model = copy.deepcopy(model)
    prunable = model.get_prunable_layers()

    # -------------------------------------------------------------------
    # Step 1: Collect all free (non-skip-locked) filter scores globally
    # -------------------------------------------------------------------
    all_scores: list[tuple[str, int, float]] = []  # (layer_name, filter_idx, score)

    for pl in prunable:
        if pl.skip_locked:
            continue
        if pl.name not in importance_scores:
            logger.warning("No importance score for layer %s — skipping.", pl.name)
            continue
        scores = importance_scores[pl.name]
        for f_idx, score in enumerate(scores.tolist()):
            all_scores.append((pl.name, f_idx, score))

    if not all_scores:
        logger.warning("No free prunable filters found. Returning model unchanged.")
        return model

    # Sort by score ascending (lowest = prune first)
    all_scores.sort(key=lambda x: x[2])

    n_total = len(all_scores)
    n_prune = int(n_total * target_ratio)
    # Safety: never prune more filters than leaves 1 per layer
    # (enforced per-layer below)
    n_prune = max(0, min(n_prune, n_total))

    prune_set: dict[str, set[int]] = {}
    for name, f_idx, _ in all_scores[:n_prune]:
        prune_set.setdefault(name, set()).add(f_idx)

    # -------------------------------------------------------------------
    # Step 2: Per-layer safety — ensure ≥1 filter is always kept
    # -------------------------------------------------------------------
    for pl in prunable:
        if pl.name not in prune_set:
            continue
        n_out = pl.conv.weight.shape[0]
        to_prune = prune_set[pl.name]
        if len(to_prune) >= n_out:
            # Keep the top-1 filter
            layer_scores = importance_scores[pl.name]
            keep_idx = int(layer_scores.argmax().item())
            to_prune.discard(keep_idx)
            logger.warning(
                "Layer %s: pruning would remove all filters — keeping filter %d.",
                pl.name, keep_idx,
            )

    # -------------------------------------------------------------------
    # Step 3: Compute kept indices per layer
    # -------------------------------------------------------------------
    kept_indices: dict[str, Tensor] = {}

    for pl in prunable:
        n_out = pl.conv.weight.shape[0]
        if pl.skip_locked or pl.name not in prune_set:
            # Keep all
            kept_indices[pl.name] = torch.arange(n_out)
        else:
            to_prune = prune_set[pl.name]
            keep = [i for i in range(n_out) if i not in to_prune]
            kept_indices[pl.name] = torch.tensor(keep, dtype=torch.long)

    # -------------------------------------------------------------------
    # Step 4: Build identity maps for module replacement
    # -------------------------------------------------------------------
    id_to_name = _build_id_to_name_map(model)
    # Map bn object id → bn name (for the *current* deep-copied model)
    bn_id_to_name: dict[int, str] = {
        id(pl.bn): id_to_name[id(pl.bn)]
        for pl in prunable
        if id(pl.bn) in id_to_name
    }
    # Build successor map BEFORE any replacements (while shapes are original)
    successor_map = _build_successor_map(model, prunable, id_to_name)
    # Remember original out_channels for successor lookup after rebuild
    orig_out_channels: dict[str, int] = {
        pl.name: pl.conv.weight.shape[0] for pl in prunable
    }

    # -------------------------------------------------------------------
    # Step 5: Rebuild each prunable conv + BN with kept output indices.
    # -------------------------------------------------------------------
    # First pass: rebuild conv and BN for output channel pruning only.
    # Key new_convs and new_bns by pl.name.
    # Also build a real_name → pl.name reverse map for successor lookup.
    new_convs: dict[str, nn.Conv2d] = {}
    new_bns: dict[str, nn.BatchNorm2d] = {}
    real_to_pl_name: dict[str, str] = {}  # real module path → pl.name

    for pl in prunable:
        new_conv = _rebuild_conv(
            pl.conv,
            kept_out_indices=kept_indices[pl.name],
            kept_in_indices=None,  # in_channels updated in second pass
        )
        new_bn = _rebuild_bn(pl.bn, kept_indices[pl.name])
        new_convs[pl.name] = new_conv
        new_bns[pl.name] = new_bn
        # Register both pl.name and real module path as keys
        real_name = id_to_name.get(id(pl.conv), pl.name)
        real_to_pl_name[real_name] = pl.name
        real_to_pl_name[pl.name] = pl.name  # also accept pl.name directly

    # Second pass: propagate input channel changes to successor layers.
    for pl in prunable:
        succs = successor_map.get(pl.name, [])
        kept = kept_indices[pl.name]
        n_kept = len(kept)

        for succ_name, succ_module in succs:
            if isinstance(succ_module, nn.Conv2d):
                is_depthwise_succ = (succ_module.groups == succ_module.in_channels)

                # Resolve succ_name to pl.name key
                succ_pl_name = real_to_pl_name.get(succ_name)
                if succ_pl_name is not None and succ_pl_name in new_convs:
                    # Successor is a prunable layer — update its in_channels
                    old_new = new_convs[succ_pl_name]
                    if is_depthwise_succ:
                        # Depthwise: slice output channels (= input channels for DW)
                        updated = _rebuild_conv(old_new, kept_out_indices=kept, kept_in_indices=None)
                    else:
                        updated = _rebuild_conv(old_new, kept_out_indices=None, kept_in_indices=kept)
                    new_convs[succ_pl_name] = updated
                else:
                    # Successor is a non-prunable conv (depthwise or shortcut projection)
                    if is_depthwise_succ:
                        # Depthwise successor: select the kept output/input channels
                        updated = _rebuild_conv(succ_module, kept_out_indices=kept, kept_in_indices=None)
                    else:
                        updated = _rebuild_conv(succ_module, kept_out_indices=None, kept_in_indices=kept)
                    _set_module_by_name(model, succ_name, updated)

                    # Also update the BN + next prunable conv after this depthwise
                    if is_depthwise_succ:
                        all_named_local = list(model.named_modules())
                        name_list = [n for n, _ in all_named_local]
                        try:
                            dw_idx = name_list.index(succ_name)
                        except ValueError:
                            dw_idx = -1

                        if dw_idx >= 0:
                            # BN: typically immediately follows the conv in the Sequential
                            dw_bn_name = None
                            for candidate_name, candidate_mod in all_named_local[dw_idx + 1:dw_idx + 3]:
                                if isinstance(candidate_mod, nn.BatchNorm2d):
                                    dw_bn_name = candidate_name
                                    break
                            if dw_bn_name:
                                old_dw_bn = dict(model.named_modules())[dw_bn_name]
                                new_dw_bn = _rebuild_bn(old_dw_bn, kept)
                                _set_module_by_name(model, dw_bn_name, new_dw_bn)

                            # Project conv: find next non-depthwise Conv2d after dw
                            # that has in_channels matching the depthwise's out_channels.
                            for proj_name, proj_mod in all_named_local[dw_idx + 1:]:
                                if isinstance(proj_mod, nn.Conv2d):
                                    if proj_mod.groups == proj_mod.in_channels:
                                        continue  # skip another depthwise
                                    # This is the project conv — update its in_channels
                                    proj_pl_name = real_to_pl_name.get(proj_name)
                                    if proj_pl_name is not None and proj_pl_name in new_convs:
                                        old_proj = new_convs[proj_pl_name]
                                        updated_proj = _rebuild_conv(
                                            old_proj,
                                            kept_out_indices=None,
                                            kept_in_indices=kept,
                                        )
                                        new_convs[proj_pl_name] = updated_proj
                                    break

            elif isinstance(succ_module, nn.Linear):
                updated_linear = _rebuild_linear_in(succ_module, kept)
                _set_module_by_name(model, succ_name, updated_linear)
                logger.debug("Updated classifier %s: in_features %d -> %d",
                             succ_name, succ_module.in_features, n_kept)

    # Third pass: install the rebuilt conv and BN modules into the model.
    # Use real module paths for installation.
    for pl in prunable:
        real_conv_name = id_to_name.get(id(pl.conv), pl.name)
        _set_module_by_name(model, real_conv_name, new_convs[pl.name])
        bn_name = bn_id_to_name.get(id(pl.bn))
        if bn_name:
            _set_module_by_name(model, bn_name, new_bns[pl.name])
        else:
            logger.warning("Could not find BN name for layer %s — BN not updated.", pl.name)

    n_pruned_total = sum(
        (orig_out_channels[pl.name] - len(kept_indices[pl.name]))
        for pl in prunable
    )
    logger.info("Pruned %d filters (target ratio=%.1f%%)", n_pruned_total, target_ratio * 100)

    return model


# ---------------------------------------------------------------------------
# _finetune_one_epoch
# ---------------------------------------------------------------------------

def _finetune_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
) -> float:
    """Run one epoch of fine-tuning and return average training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader, device) -> tuple[float, float]:
    """Evaluate model on loader. Returns (avg_loss, top1_accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total if total > 0 else float("nan")
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# iterative_prune_and_finetune
# ---------------------------------------------------------------------------

def iterative_prune_and_finetune(
    model: nn.Module,
    target_ratio: float,
    criterion_name: str,
    calibration_loader,
    train_loader,
    val_loader,
    step_ratio: float = 0.05,
    finetune_epochs: int = 2,
    finetune_lr: float = 1e-3,
    finetune_momentum: float = 0.9,
    finetune_weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    arch: str = "generic",
    input_shape: tuple = (3, 32, 32),
    seed: int = 0,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Iteratively prune + fine-tune until the target compression ratio is reached.

    Each iteration:
      1. Compute importance scores (``criterion_name``).
      2. Prune ``step_ratio`` fraction of remaining free filters.
      3. Fine-tune for ``finetune_epochs`` epochs.
      4. Log metrics.
      5. Repeat until cumulative prune ratio ≥ target_ratio.

    Args:
        model: Trained baseline model to prune.
        target_ratio: Target fraction of filters to remove (0 < r < 1).
        criterion_name: Pruning criterion — 'random', 'l1', 'taylor', 'fpgm'.
        calibration_loader: DataLoader for scoring (may equal train_loader).
        train_loader: DataLoader for fine-tuning.
        val_loader: DataLoader for validation after each step.
        step_ratio: Fraction to prune per iteration (default 0.05 = 5%).
        finetune_epochs: Epochs per fine-tuning step.
        finetune_lr: Fine-tuning learning rate.
        finetune_momentum: SGD momentum for fine-tuning.
        finetune_weight_decay: L2 regularisation for fine-tuning.
        device: Training device.
        arch: Architecture name (for logging and prune_filters).
        input_shape: (C, H, W) for FLOPs counting.
        seed: Reproducibility seed passed to criterion.
        verbose: Print per-step metrics to stdout.

    Returns:
        (pruned_model, step_logs)
        step_logs: list of dicts, one per prune iteration, with keys:
          step, cumulative_ratio_target, params, flops, val_loss, val_acc,
          finetune_loss.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be between 0 and 1 (exclusive); got {target_ratio}.")
    if not 0.0 < step_ratio < 1.0:
        raise ValueError(f"step_ratio must be between 0 and 1 (exclusive); got {step_ratio}.")
    if finetune_epochs < 1:
        raise ValueError("finetune_epochs must be at least 1.")
    model = model.to(device)
    step_logs: list[dict[str, Any]] = []

    # Initial stats
    orig_params = count_parameters(model)
    orig_flops = count_flops(model, input_shape, device)

    # Compute cumulative ratio already achieved (starts at 0)
    cumulative_pruned = 0.0
    step = 0

    if verbose:
        _, init_acc = _evaluate(model, val_loader, device)
        print(f"[Baseline] params={orig_params:,}  flops={orig_flops:,}  val_acc={init_acc:.2f}%")

    while cumulative_pruned < target_ratio - 1e-6:
        step += 1

        # How much to prune this step (don't overshoot target)
        remaining = target_ratio - cumulative_pruned
        this_step_ratio = min(step_ratio, remaining)

        # --- Score ---
        importance = compute_importance(
            model, criterion_name, calibration_loader, device, seed=seed + step
        )

        # --- Prune ---
        model = prune_filters(model, importance, this_step_ratio, arch)
        model = model.to(device)

        # Update cumulative ratio (approximate, based on target)
        cumulative_pruned = min(cumulative_pruned + this_step_ratio, target_ratio)

        # --- Fine-tune ---
        optimizer = SGD(
            model.parameters(),
            lr=finetune_lr,
            momentum=finetune_momentum,
            weight_decay=finetune_weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=finetune_epochs * len(train_loader),
        )
        criterion = nn.CrossEntropyLoss()

        ft_losses = []
        for ep in range(finetune_epochs):
            ft_loss = _finetune_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            ft_losses.append(ft_loss)

        avg_ft_loss = sum(ft_losses) / len(ft_losses)

        # --- Evaluate ---
        val_loss, val_acc = _evaluate(model, val_loader, device)
        cur_params = count_parameters(model)
        cur_flops = count_flops(model, input_shape, device)
        stats = compression_stats(orig_params, cur_params, orig_flops, cur_flops)

        log_entry = {
            "step": step,
            "cumulative_ratio_target": round(cumulative_pruned, 4),
            "params": cur_params,
            "flops": cur_flops,
            "params_reduction_pct": stats["params_reduction_pct"],
            "flops_reduction_pct": stats["flops_reduction_pct"],
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 3),
            "finetune_loss": round(avg_ft_loss, 4),
            "model_size_mb": model_size_mb(model),
        }
        step_logs.append(log_entry)

        if verbose:
            print(
                f"[Step {step}] ratio={cumulative_pruned:.0%}  "
                f"params={cur_params:,} (-{stats['params_reduction_pct']:.1f}%)  "
                f"flops={cur_flops:,} (-{stats['flops_reduction_pct']:.1f}%)  "
                f"val_acc={val_acc:.2f}%  ft_loss={avg_ft_loss:.4f}"
            )

    return model, step_logs
