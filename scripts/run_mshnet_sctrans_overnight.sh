#!/usr/bin/env bash
# 串联全量训练：先 MSHNet×4×400，再 SCTransNet×4×1000，尽量占满 GPU 不空置
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_ROOT="${DATASET_ROOT:-$ROOT/RDANet-main/datasets}"

echo "======== overnight start $(date) ========"
echo "DATASET_ROOT=$DATASET_ROOT"

bash "$ROOT/MSHNet-main/run_mshnet_all.sh"
bash "$ROOT/SCTransNet-main/run_sctrans_all.sh"

echo "======== overnight all done $(date) ========"
