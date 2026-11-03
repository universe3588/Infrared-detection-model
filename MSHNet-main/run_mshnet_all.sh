#!/usr/bin/env bash
# MSHNet 四数据集全量训练（论文默认 400 epoch）
set -euo pipefail
cd "$(dirname "$0")"

DATASET_ROOT="${DATASET_ROOT:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-400}"
BS="${BS:-4}"
LR="${LR:-0.05}"
mkdir -p logs weight

for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  echo "======== MSHNet $ds $(date) ========"
  python main.py \
    --dataset-dir "${DATASET_ROOT}/${ds}" \
    --batch-size "${BS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --warm-epoch 5 \
    --base-size 256 \
    --crop-size 256 \
    --mode train \
    --save-root ./weight \
    2>&1 | tee "logs/mshnet_${ds}_$(date +%Y%m%d_%H%M%S).log"
  echo "======== done MSHNet $ds $(date) ========"
done

echo "ALL MSHNet jobs finished at $(date)"
