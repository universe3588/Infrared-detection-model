#!/usr/bin/env bash
# 四数据集 UIU-Net 压缩流水线（需已有教师 uiunet_best.pth.tar）
set -euo pipefail
cd "$(dirname "$0")"

DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-3}"
GPUS="${GPUS:-0}"
PRUNE_RATES="${PRUNE_RATES:-0.20,0.30,0.40,0.50}"
WORK_DIR="${WORK_DIR:-./compress_runs}"

# 按本地已拉取的教师目录填写；可用环境变量覆盖
declare -A TEACHERS=(
  ["NUAA-SIRST"]="${TEACHER_NUAA:-saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar}"
  ["NUDT-SIRST"]="${TEACHER_NUDT:-saved_models/uiunet_irstd/NUDT-SIRST_20261030_164112/uiunet_best.pth.tar}"
  ["IRSTD-1k"]="${TEACHER_IRSTD:-saved_models/uiunet_irstd/IRSTD-1k_20261030_193208/uiunet_best.pth.tar}"
  ["SIRST-V2"]="${TEACHER_SIRSTV2:-saved_models/uiunet_irstd/SIRST-V2_20261030_224844/uiunet_best.pth.tar}"
)

for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ckpt="${TEACHERS[$ds]}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] missing teacher: $ckpt"
    continue
  fi
  echo "======== UIU compress $ds $(date) ========"
  python compress_train.py \
    --mode pipeline \
    --teacher_ckpt "$ckpt" \
    --dataset "$ds" \
    --dataset_dir "$DATASET_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --prune_rates "$PRUNE_RATES" \
    --gpus "$GPUS" \
    --work_dir "$WORK_DIR"
  echo "======== done $ds $(date) ========"
done

echo "ALL UIU-Net compress jobs finished at $(date)"
