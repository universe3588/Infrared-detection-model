#!/usr/bin/env bash
# 过夜串联（防空转）：
#   1) SCTransNet：NUAA / NUDT / IRSTD 统一 v3 重训（SIRST-V2 已有 v3，默认跳过）
#   2) DNANet：师生同脚本 fair re-eval（解释学生 IoU 暴涨）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
export ROOT_DATA="${ROOT_DATA:-$DATASET_DIR}"
export PYTHONUNBUFFERED=1

# 若要连 SIRST-V2 也重跑：INCLUDE_SIRSTV2=1
INCLUDE_SIRSTV2="${INCLUDE_SIRSTV2:-0}"

echo "======== overnight SCTrans-v3 + DNA fair-eval start $(date) ========"
echo "ROOT=$ROOT DATASET_DIR=$DATASET_DIR INCLUDE_SIRSTV2=$INCLUDE_SIRSTV2"

# ---------- 1) SCTrans v3 remaining ----------
cd "$ROOT/SCTransNet-main"
for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ls -lh log/${ds}/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || {
    echo "[error] missing SCTrans teacher for $ds"; exit 1;
  }
done
sed -i 's/\r$//' run_compress_all.sh run_compress_v3_all.sh 2>/dev/null || true

if [[ "$INCLUDE_SIRSTV2" == "1" ]]; then
  export DATASETS="NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2"
else
  export DATASETS="NUAA-SIRST NUDT-SIRST IRSTD-1k"
fi
bash run_compress_v3_all.sh

echo "======== SCTrans v3 stage done $(date) ========"

# ---------- 2) DNA fair re-eval ----------
cd "$ROOT/Infrared-Small-Target-Detection-master"
sed -i 's/\r$//' run_dna_fair_reeval.sh 2>/dev/null || true
# 确认教师存在
for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  ls -lh result/${ds}_DNANet_*_wDS/mIoU__DNANet_*_epoch.pth.tar 2>/dev/null | head -1 || {
    echo "[error] missing DNA teacher for $ds"; exit 1;
  }
done
# 学生可选：缺则只测教师
bash run_dna_fair_reeval.sh

echo "======== overnight ALL done $(date) ========"
echo "SCTrans log: $ROOT/SCTransNet-main/compress_v3_remaining.log"
echo "DNA reeval:  $ROOT/Infrared-Small-Target-Detection-master/compress_runs/fair_reeval_*/summary.csv"
