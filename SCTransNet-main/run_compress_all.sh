#!/usr/bin/env bash
# SCTransNet 四数据集压缩流水线（评分→剪枝→蒸馏 200 epoch）
# 用法：
#   bash run_compress_all.sh
#   bash run_compress_all.sh 2>&1 | tee compress_all.log
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-8}"
LR="${LR:-0.001}"
PRUNE_RATES="${PRUNE_RATES:-0.20,0.30,0.40,0.50}"
WORK_DIR="${WORK_DIR:-./compress_runs}"
mkdir -p "$WORK_DIR" logs

pick_teacher() {
  local ds="$1"
  local f
  f="$(ls -1t log/${ds}/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || true)"
  if [[ -z "$f" ]]; then
    echo "[error] no teacher best under log/${ds}/SCTransNet_*_best.pth.tar" >&2
    return 1
  fi
  echo "$f"
}

DATASETS=(NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2)
N=${#DATASETS[@]}
t0=$(date +%s)

for i in "${!DATASETS[@]}"; do
  ds="${DATASETS[$i]}"
  idx=$((i + 1))
  teacher="$(pick_teacher "$ds")"
  echo "======== [$idx/$N] SCTransNet compress $ds $(date) ========"
  echo "teacher=$teacher"
  python -u compress_train.py --mode pipeline \
    --teacher_ckpt "$teacher" \
    --dataset "$ds" \
    --dataset_dir "$DATASET_DIR" \
    --batch_size "$BS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --prune_rates "$PRUNE_RATES" \
    --work_dir "$WORK_DIR" \
    2>&1 | tee "logs/compress_${ds}_$(date +%Y%m%d_%H%M%S).log"

  done_n=$idx
  elapsed=$(( $(date +%s) - t0 ))
  if [[ $done_n -lt $N && $done_n -gt 0 ]]; then
    avg=$(( elapsed / done_n ))
    rem=$(( avg * (N - done_n) ))
    echo "[ETA] finished $done_n/$N, elapsed=${elapsed}s, est_remaining≈${rem}s"
  fi
  echo "======== done compress $ds $(date) ========"
done

echo "ALL SCTransNet compress jobs finished at $(date)"
