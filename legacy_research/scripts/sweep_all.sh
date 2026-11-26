#!/usr/bin/env bash
# Sequential legacy pruning sweep. Outputs remain in legacy_research/results/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
ARCHS=(resnet56 vgg16 mobilenetv2)
DATASETS=(cifar10 cifar100)
CRITERIA=(random l1 taylor fpgm)
RATIOS=(0.1 0.3 0.5 0.7 0.9)
STEP_RATIO=0.05
FINETUNE_EPOCHS=2
FINETUNE_LR=0.001
BATCH_SIZE=128
NUM_WORKERS=4
OUT="$ROOT/legacy_research/results"
DRY_RUN=0
FILTER_ARCH="" FILTER_DATASET="" FILTER_CRITERION=""
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --arch|--dataset|--criterion)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --arch) FILTER_ARCH="$2" ;;
        --dataset) FILTER_DATASET="$2" ;;
        --criterion) FILTER_CRITERION="$2" ;;
      esac
      shift ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done
total=0; skipped=0
for arch in "${ARCHS[@]}"; do
  [[ -n "$FILTER_ARCH" && "$arch" != "$FILTER_ARCH" ]] && continue
  for dataset in "${DATASETS[@]}"; do
    [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue
    ckpt="$OUT/checkpoints/${arch}_${dataset}_baseline.pt"
    if [[ ! -f "$ckpt" && $DRY_RUN -eq 0 ]]; then
      echo "[SKIP] Missing baseline checkpoint: $ckpt"; ((skipped+=1)); continue
    fi
    for criterion in "${CRITERIA[@]}"; do
      [[ -n "$FILTER_CRITERION" && "$criterion" != "$FILTER_CRITERION" ]] && continue
      for ratio in "${RATIOS[@]}"; do
        cmd=("$PYTHON" "$ROOT/legacy_research/scripts/run_baseline_pruning.py"
          --arch "$arch" --dataset "$dataset" --criterion "$criterion"
          --target_ratio "$ratio" --step_ratio "$STEP_RATIO"
          --finetune_epochs "$FINETUNE_EPOCHS" --finetune_lr "$FINETUNE_LR"
          --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
          --checkpoint_dir "$OUT/checkpoints" --results_dir "$OUT")
        if ((DRY_RUN)); then printf '[DRY-RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'
        else printf '[RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'; "${cmd[@]}"; fi
        ((total+=1))
      done
    done
  done
done
printf 'Sweep finished. Launched: %d | Skipped: %d\n' "$total" "$skipped"
