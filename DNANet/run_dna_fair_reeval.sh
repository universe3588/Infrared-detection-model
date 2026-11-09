#!/usr/bin/env bash
# DNANet：师生用 compress_train.py 同一 evaluate() / 同一 test loader 复测
# 用于回应「学生 IoU 暴涨是否因教师未充分 / 指标口径不一致」
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

ROOT_DATA="${ROOT_DATA:-../RDANet-main/datasets}"
OUT_DIR="${OUT_DIR:-./compress_runs/fair_reeval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR" logs
SUMMARY="$OUT_DIR/summary.csv"
echo "dataset,teacher_ckpt,student_ckpt,teacher_mIoU,student_mIoU,delta_pp,teacher_loss,student_loss,student_ckpt_best_iou" > "$SUMMARY"

pick_teacher() {
  local ds="$1"
  local f
  f="$(ls -1td result/${ds}_DNANet_*_wDS/mIoU__DNANet_*_epoch.pth.tar 2>/dev/null | head -1 || true)"
  if [[ -z "$f" ]]; then
    echo "[error] no teacher for $ds under result/" >&2
    return 1
  fi
  echo "$f"
}

pick_student() {
  local ds="$1"
  local f
  f="$(ls -1td compress_runs/${ds}_pipeline_*/student_best.pth.tar 2>/dev/null | head -1 || true)"
  if [[ -z "$f" ]]; then
    echo ""
    return 0
  fi
  echo "$f"
}

DATASETS=(NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2)
echo "======== DNANet fair re-eval start $(date) ========"
echo "OUT_DIR=$OUT_DIR ROOT_DATA=$ROOT_DATA"

for ds in "${DATASETS[@]}"; do
  teacher="$(pick_teacher "$ds")"
  student="$(pick_student "$ds")"
  echo "-------- $ds --------"
  echo "teacher=$teacher"
  echo "student=${student:-<none>}"
  cmd=(python -u compress_train.py --mode eval
    --teacher_ckpt "$teacher"
    --dataset "$ds"
    --root "$ROOT_DATA"
    --work_dir "$OUT_DIR"
    --deep_supervision True
    --train_aug resize
  )
  if [[ -n "$student" ]]; then
    cmd+=(--student_ckpt "$student")
  fi
  "${cmd[@]}" 2>&1 | tee "$OUT_DIR/${ds}_eval.log"

  # 从日志抽一行结果写入 csv（eval 模式也会写 json）
  json="$OUT_DIR/${ds}_eval_metrics.json"
  if [[ -f "$json" ]]; then
    python - <<PY
import json, csv
from pathlib import Path
j=json.loads(Path(r"$json").read_text(encoding="utf-8"))
row=[
  j.get("dataset",""),
  j.get("teacher_ckpt",""),
  j.get("student_ckpt",""),
  f"{j.get('teacher_mIoU', float('nan')):.6f}",
  f"{j.get('student_mIoU', float('nan')):.6f}" if j.get("student_mIoU") is not None else "",
  f"{j.get('delta_pp', float('nan')):.3f}" if j.get("delta_pp") is not None else "",
  f"{j.get('teacher_loss', float('nan')):.6f}",
  f"{j.get('student_loss', float('nan')):.6f}" if j.get("student_loss") is not None else "",
  f"{j.get('student_ckpt_best_iou', float('nan')):.6f}" if j.get("student_ckpt_best_iou") is not None else "",
]
with open(r"$SUMMARY","a",encoding="utf-8",newline="") as f:
  csv.writer(f).writerow(row)
print("[csv]", ",".join(row))
PY
  fi
done

echo "======== DNANet fair re-eval done $(date) ========"
echo "summary -> $SUMMARY"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
