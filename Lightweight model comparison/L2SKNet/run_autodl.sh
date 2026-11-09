#!/usr/bin/env bash
# AutoDL 一键：准备数据 + 训练 L2SK
# 用法：
#   bash run_autodl.sh
#   bash run_autodl.sh UNet NUAA-SIRST
#   bash run_autodl.sh FPN NUAA-SIRST NUDT-SIRST
set -euo pipefail

cd "$(dirname "$0")"

VARIANT="${1:-UNet}"
shift || true
if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(NUAA-SIRST NUDT-SIRST IRSTD-1K SIRST-V2)
fi

case "$VARIANT" in
  UNet|unet) MODEL=L2SKNet_UNet ;;
  FPN|fpn)   MODEL=L2SKNet_FPN ;;
  1D_UNet)   MODEL=L2SKNet_1D_UNet ;;
  1D_FPN)    MODEL=L2SKNet_1D_FPN ;;
  *) MODEL="$VARIANT" ;;
esac

RDA_ROOT="${RDA_ROOT:-}"
if [[ -z "$RDA_ROOT" ]]; then
  for c in /root/autodl-tmp/datasets /root/autodl-tmp/RDANet-main/datasets; do
    if [[ -d "$c" ]]; then RDA_ROOT="$c"; break; fi
  done
fi
if [[ -z "${RDA_ROOT}" ]]; then
  echo "ERROR: set RDA_ROOT=/path/to/datasets"; exit 1
fi

echo "[1/2] prepare_data --rda-root $RDA_ROOT"
python prepare_data.py --rda-root "$RDA_ROOT"

BS="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-400}"
echo "[2/2] train $MODEL on ${DATASETS[*]}  bs=$BS epochs=$EPOCHS"
python train_ex04.py \
  --model_names "$MODEL" \
  --dataset_names "${DATASETS[@]}" \
  --batchSize "$BS" \
  --nEpochs "$EPOCHS" \
  --threads "${THREADS:-4}" \
  --device "${DEVICE:-cuda:0}"

echo "done. checkpoints under ./log/<dataset>/$MODEL/best_miou.pth.tar"
