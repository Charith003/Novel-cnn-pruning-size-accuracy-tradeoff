# Pruning methods and evaluation

The preserved criteria return a per-filter importance score, with smaller scores pruned first:

- Random: seeded uniform random score.
- L1: `sum(abs(W_k))` over each output filter's weights.
- Taylor-style source baseline: `sum(abs(grad(W_k) * W_k))`. The source docstring describes activation gradients, but its implementation uses weight gradients; this module preserves that implementation and does not label it exact activation-Taylor scoring.
- FPGM: Euclidean distance from each flattened filter to a Weiszfeld estimate of the geometric median; filters closest to the median are pruned first.

Parameter reductions describe the resulting structured model. FLOPs are estimated by operator counting on a dense forward pass; do not interpret zeros from unstructured masks as dense compute savings. `model_size_mb` assumes four bytes per parameter and excludes serialization metadata and buffers.

## Taylor implementation note

The legacy Taylor function aggregates `sum(abs(gradient_weight * weight))` per filter. The source documentation describes an activation-gradient Taylor score, but the code does not compute that quantity. This implementation is preserved and the tests verify its actual weight-gradient formula.

## AMFI is separate

AMFI is not implemented or evaluated in this legacy module. Inspection found a material difference between the existing root `pruning_core.py` and the original notebook: the root core combines per-filter normalized L1, weight-gradient, activation entropy, and redundancy scores with weights `0.25`, `0.35`, `0.20`, and `-0.20`. The notebook's final AMFI scoring path uses parameter-tensor scores, global normalization (in its global mode), a layer-importance term, and coefficients defaulting to `0.25`, `0.35`, `0.25`, and `0.15`; its pruning/training path applies masks. The root implementation instead physically rebuilds convolution channels. These are not equivalent implementations, so notebook AMFI behavior or results must not be attributed to the root application. This integration changes neither implementation.

No pruning method is claimed to outperform another. Comparative conclusions require reproducible multi-seed experiments and measured size/latency where those are claimed.
