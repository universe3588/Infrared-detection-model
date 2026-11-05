#!/usr/bin/env bash
# AutoDL 过夜：MSHNet 四数据集压缩全量（pipeline ×4 ×200）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
export PYTHONUNBUFFERED=1

echo "======== MSHNet compress overnight start $(date) ========"
echo "ROOT=$ROOT DATASET_DIR=$DATASET_DIR"

cd "$ROOT/MSHNet-main"
sed -i 's/\r$//' run_compress_all.sh 2>/dev/null || true
bash run_compress_all.sh 2>&1 | tee "$ROOT/MSHNet-main/compress_all.log"

echo "======== MSHNet compress overnight done $(date) ========"
