#!/usr/bin/env bash
# 四数据集 UIU-Net 压缩流水线（需已有教师 uiunet_best.pth.tar）
# 带数据集级进度与粗略 ETA（按已完成数据集平均耗时估算）
set -euo pipefail
cd "$(dirname "$0")"

DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-3}"
GPUS="${GPUS:-0}"
PRUNE_RATES="${PRUNE_RATES:-0.20,0.30,0.40,0.50}"
WORK_DIR="${WORK_DIR:-./compress_runs}"

declare -A TEACHERS=(
  ["NUAA-SIRST"]="${TEACHER_NUAA:-saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar}"
  ["NUDT-SIRST"]="${TEACHER_NUDT:-saved_models/uiunet_irstd/NUDT-SIRST_20261030_164112/uiunet_best.pth.tar}"
  ["IRSTD-1k"]="${TEACHER_IRSTD:-saved_models/uiunet_irstd/IRSTD-1k_20261030_193208/uiunet_best.pth.tar}"
  ["SIRST-V2"]="${TEACHER_SIRSTV2:-saved_models/uiunet_irstd/SIRST-V2_20261030_224844/uiunet_best.pth.tar}"
)

DATASETS=(NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2)
TOTAL=${#DATASETS[@]}
DONE=0
ALL_T0=$(date +%s)

fmt_hms() {
  local sec=$1
  printf "%02d:%02d:%02d" $((sec/3600)) $(((sec%3600)/60)) $((sec%60))
}

echo "======== UIU compress ALL start $(date) | datasets=${TOTAL} epochs=${EPOCHS} ========"

for ds in "${DATASETS[@]}"; do
  ckpt="${TEACHERS[$ds]}"
  idx=$((DONE + 1))
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] [$idx/$TOTAL] missing teacher: $ckpt"
    DONE=$((DONE + 1))
    continue
  fi

  # ETA from finished jobs
  now=$(date +%s)
  elapsed=$((now - ALL_T0))
  if (( DONE > 0 )); then
    avg=$((elapsed / DONE))
    remain=$((avg * (TOTAL - DONE)))
    eta_msg="elapsed=$(fmt_hms "$elapsed") ETA_rest≈$(fmt_hms "$remain")"
  else
    eta_msg="elapsed=$(fmt_hms "$elapsed") ETA_rest=estimating after 1st dataset"
  fi

  echo ""
  echo ">>>>>>>> [$idx/$TOTAL] $ds | $eta_msg | $(date) <<<<<<<<"
  DS_T0=$(date +%s)
  python -u compress_train.py \
    --mode pipeline \
    --teacher_ckpt "$ckpt" \
    --dataset "$ds" \
    --dataset_dir "$DATASET_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --prune_rates "$PRUNE_RATES" \
    --gpus "$GPUS" \
    --work_dir "$WORK_DIR"
  DS_DT=$(( $(date +%s) - DS_T0 ))
  DONE=$((DONE + 1))
  echo "<<<<<<<< [$idx/$TOTAL] $ds done in $(fmt_hms "$DS_DT") | $(date) <<<<<<<<"
done

ALL_DT=$(( $(date +%s) - ALL_T0 ))
echo ""
echo "======== ALL UIU-Net compress finished in $(fmt_hms "$ALL_DT") at $(date) ========"
