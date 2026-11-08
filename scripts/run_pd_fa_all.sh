#!/usr/bin/env bash
# 一键补齐四骨干教师/学生 P_d / F_a（Ex05b / 附录 P0）
#
# 用法（在 AutoDL / 训练机上）：
#   cd /root/autodl-tmp/Infrared-detection-model   # 或你的项目根
#   bash scripts/run_pd_fa_all.sh
#
# 可选环境变量：
#   BACKBONES=dna,uiu,msh,sctrans
#   DATASETS=NUAA-SIRST,NUDT-SIRST,IRSTD-1k,SIRST-V2
#   ROLES=teacher,student
#   GPUS=0
#   MANIFEST=scripts/pd_fa_manifest.json   # 若自动发现不准则指定
#   DATASET_DIR=...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

BACKBONES="${BACKBONES:-dna,uiu,msh,sctrans}"
DATASETS="${DATASETS:-NUAA-SIRST,NUDT-SIRST,IRSTD-1k,SIRST-V2}"
ROLES="${ROLES:-teacher,student}"
GPUS="${GPUS:-0}"
DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
OUT_DIR="${OUT_DIR:-$ROOT/pd_fa_eval}"
MANIFEST="${MANIFEST:-}"

mkdir -p "$OUT_DIR" logs
LOG="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "======== P_d/F_a unified start $(date) ========"
echo "ROOT=$ROOT"
echo "DATASET_DIR=$DATASET_DIR"
echo "BACKBONES=$BACKBONES DATASETS=$DATASETS ROLES=$ROLES"
echo "OUT_DIR=$OUT_DIR"
echo "log -> $LOG"

cmd=(python -u scripts/eval_pd_fa_unified.py
  --root "$ROOT"
  --dataset_dir "$DATASET_DIR"
  --backbone "$BACKBONES"
  --datasets "$DATASETS"
  --roles "$ROLES"
  --gpus "$GPUS"
  --out_dir "$OUT_DIR"
)
if [[ -n "$MANIFEST" ]]; then
  cmd+=(--manifest "$MANIFEST")
fi

# 先 dry-run 看将评测哪些 ckpt
echo ""
echo "---- dry-run discovery ----"
"${cmd[@]}" --dry_run 2>&1 | tee -a "$LOG"

echo ""
echo "---- eval start ----"
"${cmd[@]}" 2>&1 | tee -a "$LOG"

echo ""
echo "======== P_d/F_a unified done $(date) ========"
echo "CSV/JSON/宽表: $OUT_DIR/pd_fa_*_latest.*"
ls -lh "$OUT_DIR"/pd_fa_*_latest.* 2>/dev/null || true
