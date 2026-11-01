#!/usr/bin/env bash
# SCTransNet 四数据集全量训练（论文默认 1000 epoch）
set -euo pipefail
cd "$(dirname "$0")"

DATASET_ROOT="${DATASET_ROOT:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-1000}"
BS="${BS:-16}"
GPUS="${GPUS:-0}"
mkdir -p logs log otherlogs/SCTransNet

echo "======== SCTransNet all datasets $(date) ========"
python train.py \
  --dataset_dir "${DATASET_ROOT}" \
  --dataset_names "NUAA-SIRST,NUDT-SIRST,IRSTD-1k,SIRST-V2" \
  --model_names SCTransNet \
  --epochs "${EPOCHS}" \
  --batchSize "${BS}" \
  --patchSize 256 \
  --begin_test 500 \
  --optimizer_name Adam \
  --gpus "${GPUS}" \
  --save ./log \
  2>&1 | tee "logs/sctrans_all_$(date +%Y%m%d_%H%M%S).log"

echo "ALL SCTransNet jobs finished at $(date)"
