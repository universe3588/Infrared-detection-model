#!/usr/bin/env bash
# SCTransNet 统一 v3 配额全量（或剩余三集）
# 强制：PRUNE_RATES=0.05,0.15,0.25,0.35  EPOCHS=350  LR=3e-4  LAMBDA_REGION=0.75
#
# 用法：
#   bash run_compress_v3_all.sh                         # 默认只跑 NUAA/NUDT/IRSTD（SIRST-V2 已有 v3）
#   DATASETS="NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2" bash run_compress_v3_all.sh  # 四集全量
set -euo pipefail
cd "$(dirname "$0")"

export DATASETS="${DATASETS:-NUAA-SIRST NUDT-SIRST IRSTD-1k}"
export PRUNE_RATES="0.05,0.15,0.25,0.35"
export EPOCHS="${EPOCHS:-350}"
export LR="${LR:-0.0003}"
export LAMBDA_REGION="${LAMBDA_REGION:-0.75}"
export WORK_DIR="${WORK_DIR:-./compress_runs}"

echo "======== SCTransNet unified v3 $(date) ========"
echo "DATASETS=$DATASETS"
echo "PRUNE_RATES=$PRUNE_RATES EPOCHS=$EPOCHS LR=$LR LAMBDA_REGION=$LAMBDA_REGION"
bash run_compress_all.sh 2>&1 | tee compress_v3_remaining.log
echo "======== SCTransNet unified v3 done $(date) ========"
grep -E 'best_iou=|student channel_num|params=|done compress|ALL ' compress_v3_remaining.log | grep -v 'train \|eval  ' | tail -40 || true
