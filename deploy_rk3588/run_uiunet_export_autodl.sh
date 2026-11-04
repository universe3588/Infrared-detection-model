#!/usr/bin/env bash
# AutoDL：UIU-Net 教师/学生 → ONNX + Params/FLOPs（NUAA @256，对齐 DNANet 板端协议）
# 用法：
#   cd ~/autodl-tmp/Infrared-detection-model
#   bash deploy_rk3588/run_uiunet_export_autodl.sh
#
# 依赖：torch；可选 thop、rknn-toolkit2
# 注意：compress_runs 下需有 student_best.pth.tar（勿只同步了 distill_log）

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/deploy_rk3588"

TEACHER="${TEACHER_CKPT:-$ROOT/IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar}"
STUDENT="${STUDENT_CKPT:-}"
INPUT_SIZE="${INPUT_SIZE:-256}"
OUT_DIR="outputs/uiunet"
mkdir -p "$OUT_DIR"

if [[ -z "$STUDENT" ]]; then
  # 取 NUAA 最新 pipeline 目录下的 student_best
  STUDENT="$(ls -1t "$ROOT"/IEEE_TIP_UIU-Net-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar 2>/dev/null | head -1 || true)"
fi

echo "ROOT=$ROOT"
echo "TEACHER=$TEACHER"
echo "STUDENT=$STUDENT"
echo "INPUT_SIZE=$INPUT_SIZE"

[[ -f "$TEACHER" ]] || { echo "[error] teacher missing: $TEACHER"; exit 1; }
[[ -f "$STUDENT" ]] || {
  echo "[error] student_best.pth.tar missing."
  echo "  请确认 AutoDL 上 compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar 存在"
  echo "  本地若只有 distill_log/args.json，需从云端把权重同步下来再跑。"
  exit 1
}

echo "======== 1) export ONNX ========"
python export_uiunet_onnx.py --ckpt "$TEACHER" --out "$OUT_DIR/teacher_nuaa.onnx" --input_size "$INPUT_SIZE"
python export_uiunet_onnx.py --ckpt "$STUDENT" --out "$OUT_DIR/student_nuaa.onnx" --input_size "$INPUT_SIZE"

echo "======== 2) Params / FLOPs ========"
python measure_uiunet_complexity.py \
  --teacher_ckpt "$TEACHER" \
  --student_ckpt "$STUDENT" \
  --input_size "$INPUT_SIZE" \
  --out "$OUT_DIR/complexity_nuaa.csv"

echo "======== 3) optional RKNN (needs rknn-toolkit2) ========"
if python -c "from rknn.api import RKNN" 2>/dev/null; then
  python convert_rknn.py --onnx "$OUT_DIR/teacher_nuaa.onnx" --out "$OUT_DIR/teacher_nuaa.rknn"
  python convert_rknn.py --onnx "$OUT_DIR/student_nuaa.onnx" --out "$OUT_DIR/student_nuaa.rknn"
  ls -lh "$OUT_DIR"/*.rknn
else
  echo "[skip] rknn-toolkit2 未安装；请在有 toolkit2 的环境转 RKNN，或把 ONNX 拷走再转"
fi

echo "======== done ========"
ls -lh "$OUT_DIR"
echo "下一步：把 $OUT_DIR/*.rknn（或 .onnx）拉到本机，按 ADB_DEPLOY_UIUNET.md 上板测 resource_bench"
