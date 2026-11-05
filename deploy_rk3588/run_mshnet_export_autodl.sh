#!/usr/bin/env bash
# AutoDL：MSHNet 教师/学生 → ONNX + Params/FLOPs（NUAA @256）
# 用法：
#   cd ~/autodl-tmp/Infrared-detection-model
#   bash deploy_rk3588/run_mshnet_export_autodl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/deploy_rk3588"

TEACHER="${TEACHER_CKPT:-}"
STUDENT="${STUDENT_CKPT:-}"
INPUT_SIZE="${INPUT_SIZE:-256}"
OUT_DIR="outputs/mshnet"
mkdir -p "$OUT_DIR"

if [[ -z "$TEACHER" ]]; then
  TEACHER="$(ls -1t "$ROOT"/MSHNet-main/weight/MSHNet-NUAA-SIRST-*/weight.pkl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$STUDENT" ]]; then
  STUDENT="$(ls -1t "$ROOT"/MSHNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar 2>/dev/null | head -1 || true)"
fi

echo "ROOT=$ROOT"
echo "TEACHER=$TEACHER"
echo "STUDENT=$STUDENT"
echo "INPUT_SIZE=$INPUT_SIZE"

[[ -f "$TEACHER" ]] || { echo "[error] teacher missing: $TEACHER"; exit 1; }
[[ -f "$STUDENT" ]] || {
  echo "[error] student_best.pth.tar missing under MSHNet-main/compress_runs/"
  exit 1
}

echo "======== 1) export ONNX ========"
python export_mshnet_onnx.py --ckpt "$TEACHER" --out "$OUT_DIR/teacher_nuaa.onnx" --input_size "$INPUT_SIZE"
python export_mshnet_onnx.py --ckpt "$STUDENT" --out "$OUT_DIR/student_nuaa.onnx" --input_size "$INPUT_SIZE"

echo "======== 2) Params / FLOPs ========"
python measure_mshnet_complexity.py \
  --teacher_ckpt "$TEACHER" \
  --student_ckpt "$STUDENT" \
  --input_size "$INPUT_SIZE" \
  --out "$OUT_DIR/complexity_nuaa.csv"

echo "======== 3) optional RKNN ========"
if python -c "from rknn.api import RKNN" 2>/dev/null; then
  python convert_rknn.py --onnx "$OUT_DIR/teacher_nuaa.onnx" --out "$OUT_DIR/teacher_nuaa.rknn"
  python convert_rknn.py --onnx "$OUT_DIR/student_nuaa.onnx" --out "$OUT_DIR/student_nuaa.rknn"
  ls -lh "$OUT_DIR"/*.rknn
else
  echo "[skip] rknn-toolkit2 未安装"
fi

echo "======== done ========"
ls -lh "$OUT_DIR"
echo "拉回 outputs/mshnet/*.rknn 与 complexity_nuaa.csv，按 ADB_DEPLOY_MSHNET.md 上板"
