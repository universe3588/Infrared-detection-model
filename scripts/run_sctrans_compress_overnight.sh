#!/usr/bin/env bash
# AutoDL：SCTransNet 压缩（默认难集优先；可 FULL=1 跑四集）
# 框架仍为 P1→P2→P3；仅本骨干用更保守 {p_s} + 更长蒸馏。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
export PYTHONUNBUFFERED=1
export PRUNE_RATES="${PRUNE_RATES:-0.10,0.20,0.30,0.40}"
export EPOCHS="${EPOCHS:-300}"
export LR="${LR:-0.0005}"
export LAMBDA_REGION="${LAMBDA_REGION:-0.75}"

echo "======== SCTransNet compress overnight start $(date) ========"
echo "ROOT=$ROOT DATASET_DIR=$DATASET_DIR FULL=${FULL:-0}"
echo "PRUNE_RATES=$PRUNE_RATES EPOCHS=$EPOCHS LR=$LR LAMBDA_REGION=$LAMBDA_REGION"

cd "$ROOT/SCTransNet-main"
for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ls -lh log/${ds}/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || {
    echo "[error] missing teacher for $ds"; exit 1;
  }
done

sed -i 's/\r$//' run_compress_all.sh run_compress_hardsets.sh 2>/dev/null || true

if [[ "${FULL:-0}" == "1" ]]; then
  export DATASETS="NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2"
  bash run_compress_all.sh 2>&1 | tee "$ROOT/SCTransNet-main/compress_all_v2.log"
else
  # 默认只救难集，省时间；NUAA/NUDT 首轮已可接受
  bash run_compress_hardsets.sh
fi

echo "======== SCTransNet compress overnight done $(date) ========"
