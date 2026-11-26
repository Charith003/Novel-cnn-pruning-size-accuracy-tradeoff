# Experiment guide

Keep source datasets outside this repository or under an ignored local data path. Train a baseline checkpoint, then invoke `scripts/run_baseline_pruning.py` with the architecture, dataset, method and ratios. Outputs are isolated under `legacy_research/results/`; each pruning output gets a timestamped name and exclusive creation. The sweep supports `--dry-run` and filters for architecture, dataset and criterion; failed commands stop the sweep with a nonzero exit status.

The nine source YAML files remain unchanged. They describe training settings; the legacy CLI currently accepts command-line options rather than loading YAML. Record the actual arguments and seed with each experiment. Do not compare estimated size/FLOPs with actual serialized size/latency without measuring those quantities separately.
