# Legacy CNN pruning research module

This is an experimental, isolated copy of reusable code from the older CNN filter pruning project. It is separate from the root AMFI implementation and is not a replacement for it. Nothing here is imported by the main application.

## Contents and scope

- Architectures: CIFAR variants of ResNet-56, VGG-16-BN, and MobileNetV2. They expect RGB images, conventionally 32×32, and return logits with the configured class count.
- Baselines: random, L1 filter magnitude, Taylor-style weight-gradient × weight, and FPGM geometric-median distance. The Taylor source implementation uses `|gradient_weight * weight|` aggregated per filter; despite its docstring's activation wording, it does not calculate the exact activation-gradient Taylor score. This behavior is preserved and explicitly limited here.
- Data: CIFAR-10, CIFAR-100 and custom ImageFolder folders, with train/validation separation. CIFAR downloads require the explicit `--download` option; custom data paths must be supplied. No large data is included.
- Metrics: parameter counts, estimated dense FLOPs, parameter/model-size estimates, compression ratio and reductions. FLOPs count dense operators; unstructured masking does not imply lower dense FLOPs. Model size is an estimate from parameter count and dtype, not a serialized checkpoint measurement.
- `train.py` and `prune.py` are isolated legacy entry points. Outputs default to `legacy_research/results/`; pruning outputs use unique names, and baseline training refuses to overwrite existing output files.

## Install and run

From the repository root, install only into a separate environment:

```sh
python -m venv legacy_research/.venv
# activate that environment, then:
python -m pip install -r legacy_research/requirements-legacy.txt
```

The original project metadata did not pin Python or PyTorch versions; its requirements specify `torch>=2.0.0` and `torchvision>=0.15.0`. Select mutually compatible versions for your platform. Existing root requirements are not changed.

Train a baseline, then prune it:

```sh
python -m legacy_research.train --help
python legacy_research/scripts/run_baseline_pruning.py --arch resnet56 --dataset cifar10 --criterion l1 --target_ratio 0.5 --seed 42
bash legacy_research/scripts/sweep_all.sh --dry-run --dataset cifar10
```

The nine YAML files in `configs/` are preserved training configurations from the source. The legacy CLI does not parse YAML; pass the corresponding values as CLI options. `data_root` and custom ImageFolder paths are local inputs, and missing datasets/checkpoints need to be supplied by the user. Dataset acquisition is not run as part of integration.

Run the tests with the standard-library runner: `python -m unittest discover -s legacy_research/tests -v`. Tests use synthetic data; CIFAR constructors are mocked, and no test downloads CIFAR. `pytest` is not included in the legacy requirements.

## Reproducibility and limitations

The YAML files contain training settings only; neither CLI currently loads YAML. Their relative data/checkpoint/result paths are preserved as source values, so pass paths explicitly on the command line when needed. Pruning method and target ratio are CLI arguments, not fields in these training YAML files.

Set `--seed` and record architecture, dataset, criterion, target/step ratio, and runtime. Results record the seed and elapsed time. GPU kernels and dependency/platform variation can still affect bitwise reproducibility. FLOPs are estimates and the optional `thop` counter may not cover every operator. No comparative accuracy or performance claims are made; reliable comparisons need repeated multi-seed experiments.

## Validation status

The isolated module has been smoke-tested on CPU with synthetic inputs: all three model forward passes, the four legacy importance criteria, one iterative prune/fine-tune/evaluation path, metric helpers, and synthetic dataset construction. This is functional smoke validation, not a full research reproduction. CUDA and full CIFAR training were not tested. The integrated legacy module does not implement AMFI. The repository's root AMFI code and the AMFI notebook also differ; see `docs/PRUNING_METHODS.md` before interpreting AMFI results.

Pruned checkpoints include a serialized model object to retain the physically pruned topology. `load_pruned_checkpoint` uses pickle-based loading; load only checkpoint files produced by a trusted local run. Training refuses to overwrite existing baseline checkpoint or log files; choose fresh output directories for a new baseline run.
