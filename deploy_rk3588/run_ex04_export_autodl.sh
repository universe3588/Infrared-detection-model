#!/usr/bin/env bash
# AutoDL：导出 L2SK / RepISD ONNX + 统计复杂度；可选转 RKNN
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/ex04

L2SK_CKPT="${L2SK_CKPT:-../Lightweight model comparison/L2SKNet/log/NUAA-SIRST/L2SKNet_UNet/best_miou.pth.tar}"
REP_CKPT="${REP_CKPT:-../Lightweight model comparison/RepISD-Net/result/NUAA/deploy.pth.tar}"

echo "[1] complexity"
python measure_ex04_complexity.py --l2sk-ckpt "$L2SK_CKPT" --repisd-ckpt "$REP_CKPT"

echo "[2] export L2SK ONNX (1ch@256)"
python export_l2sk_onnx.py --ckpt "$L2SK_CKPT" --out outputs/ex04/l2sk_unet_nuaa.onnx

echo "[3] export RepISD ONNX (3ch@256)"
python export_repisd_onnx.py --ckpt "$REP_CKPT" --out outputs/ex04/repisd_nuaa.onnx

if python -c "import rknn" 2>/dev/null; then
  echo "[4] convert RKNN"
  # L2SK 单通道
  python convert_rknn.py \
    --onnx outputs/ex04/l2sk_unet_nuaa.onnx \
    --out outputs/ex04/l2sk_unet_nuaa.rknn \
    --mean 0 --std 255
  # RepISD 三通道（训练归一化近似用 0/255；精确对齐可再改）
  python convert_rknn.py \
    --onnx outputs/ex04/repisd_nuaa.onnx \
    --out outputs/ex04/repisd_nuaa.rknn \
    --mean 0 0 0 --std 255 255 255
else
  echo "[4] skip RKNN（未安装 rknn-toolkit2）。把 ONNX 拷到有 toolkit2 的环境再转。"
fi

echo "done -> outputs/ex04/"
ls -lh outputs/ex04/
