#!/usr/bin/env bash
# 检查 UIU 学生权重是否齐全（板端导出前置）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "===== teacher (NUAA) ====="
ls -lh "$ROOT"/IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd/NUAA-SIRST_*/uiunet_best.pth.tar 2>/dev/null || echo MISSING
echo "===== student_best all datasets ====="
ls -lh "$ROOT"/IEEE_TIP_UIU-Net-main/compress_runs/*/student_best.pth.tar 2>/dev/null || echo "MISSING student_best — 不可导出学生 ONNX"
echo "===== keep_masks / scores (optional) ====="
ls -lh "$ROOT"/IEEE_TIP_UIU-Net-main/compress_runs/*/keep_masks.pt 2>/dev/null || true
du -sh "$ROOT"/IEEE_TIP_UIU-Net-main/compress_runs/*/ 2>/dev/null || true
