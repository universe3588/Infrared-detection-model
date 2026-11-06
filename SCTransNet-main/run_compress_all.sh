#!/usr/bin/env bash
# SCTransNet 压缩流水线（P1→P2→P3，框架不变）
#
# 相对首轮更保守的默认（仅本骨干）：
#   PRUNE_RATES=0.10,0.20,0.30,0.40
#   EPOCHS=300  LR=5e-4  LAMBDA_REGION=0.75
#
# 用法：
#   bash run_compress_all.sh                          # 四数据集
#   DATASETS="IRSTD-1k SIRST-V2" bash run_compress_all.sh   # 只重跑难集
#   bash run_compress_all.sh 2>&1 | tee compress_all_v2.log
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-300}"
BS="${BS:-8}"
LR="${LR:-0.0005}"
MIN_LR="${MIN_LR:-0.00001}"
PRUNE_RATES="${PRUNE_RATES:-0.10,0.20,0.30,0.40}"
LAMBDA_REGION="${LAMBDA_REGION:-0.75}"
LAMBDA_KD="${LAMBDA_KD:-1.0}"
ALPHA="${ALPHA:-0.7}"
WORK_DIR="${WORK_DIR:-./compress_runs}"
# 空格分隔；默认四集。难集优先可：DATASETS="IRSTD-1k SIRST-V2"
DATASETS_STR="${DATASETS:-NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2}"
# shellcheck disable=SC2206
DATASETS=($DATASETS_STR)

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

N=${#DATASETS[@]}
t0=$(date +%s)

echo "[config] datasets=${DATASETS[*]}"
echo "[config] prune_rates=$PRUNE_RATES epochs=$EPOCHS lr=$LR lambda_region=$LAMBDA_REGION"
echo "[config] work_dir=$WORK_DIR  (framework=P1/P2/P3 unchanged)"

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
    --min_lr "$MIN_LR" \
    --prune_rates "$PRUNE_RATES" \
    --lambda_region "$LAMBDA_REGION" \
    --lambda_kd "$LAMBDA_KD" \
    --alpha "$ALPHA" \
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
