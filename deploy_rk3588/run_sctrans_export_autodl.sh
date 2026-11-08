#!/usr/bin/env bash
# AutoDL：SCTransNet 教师/学生 → ONNX + Params/FLOPs（+ 可选 RKNN）
# 协议：NUAA-SIRST，256×256，输入 1 通道（与训练一致）
#
# 用法：
#   cd ~/autodl-tmp/Infrared-detection-model
#   # 确认 v3 学生权重已在 SCTransNet-main/compress_runs/
#   sed -i 's/\r$//' deploy_rk3588/run_sctrans_export_autodl.sh
#   bash deploy_rk3588/run_sctrans_export_autodl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/deploy_rk3588"

TEACHER="${TEACHER_CKPT:-}"
STUDENT="${STUDENT_CKPT:-}"
INPUT_SIZE="${INPUT_SIZE:-256}"
IN_CH="${IN_CH:-1}"
OUT_DIR="outputs/sctrans"
mkdir -p "$OUT_DIR"

if [[ -z "$TEACHER" ]]; then
  TEACHER="$(ls -1t "$ROOT"/SCTransNet-main/log/NUAA-SIRST/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || true)"
fi
if [[ -z "$STUDENT" ]]; then
  # 优先带 channel_num 的 v3 学生；按时间取最新 NUAA pipeline
  STUDENT="$(ls -1t "$ROOT"/SCTransNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar 2>/dev/null | head -1 || true)"
fi

echo "ROOT=$ROOT"
echo "TEACHER=$TEACHER"
echo "STUDENT=$STUDENT"
echo "INPUT_SIZE=$INPUT_SIZE IN_CH=$IN_CH"

[[ -f "$TEACHER" ]] || { echo "[error] teacher missing: $TEACHER"; exit 1; }
[[ -f "$STUDENT" ]] || {
  echo "[error] student_best.pth.tar missing under SCTransNet-main/compress_runs/"
  echo "请先把 NUAA v3 的 compress_runs/.../student_best.pth.tar 放到服务器"
  exit 1
}

echo "======== 1) export ONNX (1-ch) ========"
python export_sctrans_onnx.py --ckpt "$TEACHER" --out "$OUT_DIR/teacher_nuaa.onnx" \
  --input_size "$INPUT_SIZE" --in_channels "$IN_CH"
python export_sctrans_onnx.py --ckpt "$STUDENT" --out "$OUT_DIR/student_nuaa.onnx" \
  --input_size "$INPUT_SIZE" --in_channels "$IN_CH"

echo "======== 2) Params / FLOPs ========"
python measure_sctrans_complexity.py \
  --teacher_ckpt "$TEACHER" \
  --student_ckpt "$STUDENT" \
  --input_size "$INPUT_SIZE" \
  --in_channels "$IN_CH" \
  --out "$OUT_DIR/complexity_nuaa.csv"

echo "======== 3) optional RKNN (1-ch mean/std) ========"
if python -c "from rknn.api import RKNN" 2>/dev/null; then
  python convert_rknn.py --onnx "$OUT_DIR/teacher_nuaa.onnx" --out "$OUT_DIR/teacher_nuaa.rknn" \
    --mean 0 --std 255
  python convert_rknn.py --onnx "$OUT_DIR/student_nuaa.onnx" --out "$OUT_DIR/student_nuaa.rknn" \
    --mean 0 --std 255
  ls -lh "$OUT_DIR"/*.rknn
else
  echo "[skip] rknn-toolkit2 未安装；可在本机 convert 或安装后重跑本脚本第 3 步"
fi

echo "======== done ========"
ls -lh "$OUT_DIR"
cat "$OUT_DIR/complexity_nuaa.csv"
echo
echo "下一步：拉回 outputs/sctrans/*.rknn 与 complexity_nuaa.csv，按 ADB_DEPLOY_SCTRANS.md 上板"
