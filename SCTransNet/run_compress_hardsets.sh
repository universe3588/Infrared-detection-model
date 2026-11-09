#!/usr/bin/env bash
# 仅重跑 IRSTD-1k / SIRST-V2（首轮掉点最大），框架不变、配额更温和。
# DNA / UIU / MSH 无需重训。
set -euo pipefail
cd "$(dirname "$0")"

export DATASETS="IRSTD-1k SIRST-V2"
export PRUNE_RATES="${PRUNE_RATES:-0.10,0.20,0.30,0.40}"
export EPOCHS="${EPOCHS:-300}"
export LR="${LR:-0.0005}"
export LAMBDA_REGION="${LAMBDA_REGION:-0.75}"
export WORK_DIR="${WORK_DIR:-./compress_runs}"

echo "======== SCTrans hard-set recovery $(date) ========"
bash run_compress_all.sh 2>&1 | tee compress_hardsets_v2.log
echo "======== hard-set recovery done $(date) ========"
