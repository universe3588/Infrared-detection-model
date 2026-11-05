#!/usr/bin/env bash
# AutoDL 过夜：SCTransNet 四数据集压缩全量（pipeline ×4 ×200）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
export PYTHONUNBUFFERED=1

echo "======== SCTransNet compress overnight start $(date) ========"
echo "ROOT=$ROOT DATASET_DIR=$DATASET_DIR"

cd "$ROOT/SCTransNet-main"
# 确认教师权重
for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ls -lh log/${ds}/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || {
    echo "[error] missing teacher for $ds"; exit 1;
  }
done

sed -i 's/\r$//' run_compress_all.sh 2>/dev/null || true
bash run_compress_all.sh 2>&1 | tee "$ROOT/SCTransNet-main/compress_all.log"

echo "======== SCTransNet compress overnight done $(date) ========"
