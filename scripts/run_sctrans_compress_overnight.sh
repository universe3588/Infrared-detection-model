#!/usr/bin/env bash
# AutoDL：SCTransNet 压缩（默认难集优先；可 FULL=1 跑四集）
# 框架仍为 P1→P2→P3；仅本骨干用更保守 {p_s} + 更长蒸馏。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
export PYTHONUNBUFFERED=1
# MODE:
#   v3_remaining（默认）= NUAA+NUDT+IRSTD 统一 v3，然后交给串联脚本也可
#   v3_all | sirstv2_v3 | hard_v2 | full
MODE="${MODE:-v3_remaining}"

echo "======== SCTransNet compress overnight start $(date) ========"
echo "ROOT=$ROOT DATASET_DIR=$DATASET_DIR MODE=$MODE FULL=${FULL:-0}"

cd "$ROOT/SCTransNet-main"
for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ls -lh log/${ds}/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || {
    echo "[error] missing teacher for $ds"; exit 1;
  }
done

sed -i 's/\r$//' run_compress_all.sh run_compress_hardsets.sh run_compress_sirstv2_v3.sh run_compress_v3_all.sh 2>/dev/null || true

if [[ "$MODE" == "full" || "$MODE" == "v3_all" || "${FULL:-0}" == "1" ]]; then
  export DATASETS="NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2"
  bash run_compress_v3_all.sh
elif [[ "$MODE" == "hard_v2" ]]; then
  bash run_compress_hardsets.sh
elif [[ "$MODE" == "sirstv2_v3" ]]; then
  bash run_compress_sirstv2_v3.sh
else
  # 默认：剩余三集统一 v3（SIRST-V2 已完成）
  bash run_compress_v3_all.sh
fi

echo "======== SCTransNet compress overnight done $(date) ========"
