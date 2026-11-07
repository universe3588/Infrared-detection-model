#!/usr/bin/env bash
# AutoDL：SCTransNet 压缩（默认难集优先；可 FULL=1 跑四集）
# 框架仍为 P1→P2→P3；仅本骨干用更保守 {p_s} + 更长蒸馏。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
export PYTHONUNBUFFERED=1
# MODE: sirstv2_v3（默认，冲 ≤3pp）| hard_v2 | full
MODE="${MODE:-sirstv2_v3}"

echo "======== SCTransNet compress overnight start $(date) ========"
echo "ROOT=$ROOT DATASET_DIR=$DATASET_DIR MODE=$MODE FULL=${FULL:-0}"

cd "$ROOT/SCTransNet-main"
for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ls -lh log/${ds}/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || {
    echo "[error] missing teacher for $ds"; exit 1;
  }
done

sed -i 's/\r$//' run_compress_all.sh run_compress_hardsets.sh run_compress_sirstv2_v3.sh 2>/dev/null || true

if [[ "$MODE" == "full" || "${FULL:-0}" == "1" ]]; then
  export PRUNE_RATES="${PRUNE_RATES:-0.10,0.20,0.30,0.40}"
  export EPOCHS="${EPOCHS:-300}"
  export LR="${LR:-0.0005}"
  export LAMBDA_REGION="${LAMBDA_REGION:-0.75}"
  export DATASETS="NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2"
  bash run_compress_all.sh 2>&1 | tee "$ROOT/SCTransNet-main/compress_all_v2.log"
elif [[ "$MODE" == "hard_v2" ]]; then
  bash run_compress_hardsets.sh
else
  # 默认：SIRST-V2 v3（脚本内强制 0.05,0.15,0.25,0.35）
  bash run_compress_sirstv2_v3.sh
fi

echo "======== SCTransNet compress overnight done $(date) ========"
