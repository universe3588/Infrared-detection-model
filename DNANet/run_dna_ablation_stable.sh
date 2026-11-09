#!/usr/bin/env bash
# DNANet 更稳消融方案：
#   1) NUAA：Group A（蒸馏恢复）+ Group B（准则对照）→ 填论文两表
#   2) NUDT：仅 Group A（回应合成集上学生冲到 0.92 的质疑）
#
# 用法：
#   bash run_dna_ablation_stable.sh
#   SKIP_NUDT=1 bash run_dna_ablation_stable.sh   # 只要 NUAA
#   SKIP_B=1 bash run_dna_ablation_stable.sh       # 跳过准则组（更省时间）
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

ROOT_DATA="${ROOT_DATA:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-8}"
PRUNE_RATES="${PRUNE_RATES:-0.20,0.30,0.40,0.50,0.60}"
SKIP_NUDT="${SKIP_NUDT:-0}"
SKIP_B="${SKIP_B:-0}"
WORK_ROOT="${WORK_ROOT:-./compress_runs/ablation_stable_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$WORK_ROOT" logs

pick_teacher() {
  local ds="$1"
  ls -1td result/${ds}_DNANet_*_wDS/mIoU__DNANet_*_epoch.pth.tar 2>/dev/null | head -1 || true
}

run_group_A() {
  local DS="$1"
  local TEACHER="$2"
  local A_DIR="$3"
  local TEACHER_REF="$4"
  mkdir -p "$A_DIR"

  local COMMON=(
    --teacher_ckpt "$TEACHER"
    --dataset "$DS"
    --root "$ROOT_DATA"
    --deep_supervision True
    --train_aug resize
    --train_batch_size "$BS"
    --test_batch_size "$BS"
    --prune_rates "$PRUNE_RATES"
    --epochs "$EPOCHS"
    --lr 0.05
    --optimizer Adagrad
    --seed 42
    --workers 4
  )

  echo "======== [$DS] Group A distill start $(date) ========"
  echo "TEACHER=$TEACHER  REF_mIoU=$TEACHER_REF"

  echo "---- [$DS][A0] full pipeline (shared masks + A4 distill) ----"
  python -u compress_train.py --mode pipeline \
    "${COMMON[@]}" \
    --prune_criterion full --alpha 0.7 \
    --lambda_seg 1.0 --lambda_region 0.5 --lambda_kd 1.0 \
    --work_dir "$A_DIR" \
    2>&1 | tee "$A_DIR/A0_full_pipeline.log"

  local BASE_PIPE KEEP INIT
  BASE_PIPE="$(ls -1td "$A_DIR"/${DS}_pipeline_* 2>/dev/null | head -1)"
  KEEP="$BASE_PIPE/keep_masks.pt"
  INIT="$BASE_PIPE/student_init.pth.tar"
  echo "BASE_PIPE=$BASE_PIPE" | tee "$A_DIR/BASE_PIPE.txt"

  echo "---- [$DS][A1] prune w/o retrain ----"
  python -u compress_train.py --mode eval \
    --teacher_ckpt "$TEACHER" \
    --student_ckpt "$INIT" \
    --dataset "$DS" --root "$ROOT_DATA" \
    --deep_supervision True --train_aug resize \
    --work_dir "$A_DIR/A1_no_retrain" \
    2>&1 | tee "$A_DIR/A1_no_retrain.log"
  cp -f "$A_DIR/A1_no_retrain/${DS}_eval_metrics.json" "$A_DIR/A1_no_retrain_metrics.json" 2>/dev/null || true

  run_distill_variant() {
    local name="$1" seg="$2" region="$3" kd="$4"
    local out="$A_DIR/$name"
    mkdir -p "$out"
    echo "---- [$DS][$name] seg=$seg region=$region kd=$kd ----"
    python -u compress_train.py --mode distill \
      "${COMMON[@]}" \
      --keep_path "$KEEP" \
      --student_ckpt "$INIT" \
      --lambda_seg "$seg" --lambda_region "$region" --lambda_kd "$kd" \
      --work_dir "$out" \
      2>&1 | tee "$out/train.log"
  }

  run_distill_variant "A2_seg_only" 1.0 0.0 0.0
  run_distill_variant "A3_seg_region" 1.0 0.5 0.0
  echo "[$DS] A4_full = $BASE_PIPE (from A0)"

  # 本组小汇总
  {
    echo "[$DS] Group A $(date)"
    echo "teacher=$TEACHER  teacher_fair_ref=$TEACHER_REF"
    if [[ -f "$A_DIR/A1_no_retrain_metrics.json" ]]; then
      python -c "import json; j=json.load(open(r'$A_DIR/A1_no_retrain_metrics.json')); print(f\"A1_no_retrain student_mIoU={j.get('student_mIoU')} teacher_mIoU={j.get('teacher_mIoU')}\")"
    fi
    echo -n "A2_seg_only: "; grep -E 'best mIoU=|best student mIoU=' "$A_DIR/A2_seg_only/train.log" 2>/dev/null | tail -1 || true
    echo -n "A3_seg_region: "; grep -E 'best mIoU=|best student mIoU=' "$A_DIR/A3_seg_region/train.log" 2>/dev/null | tail -1 || true
    echo -n "A4_full(A0): "; grep -E 'best mIoU=|best student mIoU=' "$A_DIR/A0_full_pipeline.log" 2>/dev/null | tail -1 || true
  } | tee "$A_DIR/summary_A.txt"
}

run_group_B_nuaa() {
  local DS="NUAA-SIRST"
  local TEACHER="$1"
  local B_DIR="$2"
  local A0_LOG="$3"
  mkdir -p "$B_DIR"

  local COMMON=(
    --teacher_ckpt "$TEACHER"
    --dataset "$DS"
    --root "$ROOT_DATA"
    --deep_supervision True
    --train_aug resize
    --train_batch_size "$BS"
    --test_batch_size "$BS"
    --prune_rates "$PRUNE_RATES"
    --epochs "$EPOCHS"
    --lr 0.05
    --optimizer Adagrad
    --seed 42
    --workers 4
  )

  echo "======== [$DS] Group B criterion start $(date) ========"
  for crit in random l1 task contrast; do
    echo "---- [B] criterion=$crit ----"
    python -u compress_train.py --mode pipeline \
      "${COMMON[@]}" \
      --prune_criterion "$crit" --alpha 0.7 \
      --lambda_seg 1.0 --lambda_region 0.5 --lambda_kd 1.0 \
      --work_dir "$B_DIR/$crit" \
      2>&1 | tee "$B_DIR/${crit}.log"
  done

  {
    echo "[NUAA] Group B $(date)"
    echo -n "full(from A0): "; grep -E 'best mIoU=|best student mIoU=' "$A0_LOG" 2>/dev/null | tail -1 || true
    for crit in random l1 task contrast; do
      echo -n "$crit: "; grep -E 'best mIoU=|best student mIoU=' "$B_DIR/${crit}.log" 2>/dev/null | tail -1 || true
    done
  } | tee "$B_DIR/summary_B.txt"
}

# ========================= main =========================
echo "======== DNA ablation STABLE start $(date) ========"
echo "WORK_ROOT=$WORK_ROOT EPOCHS=$EPOCHS SKIP_NUDT=$SKIP_NUDT SKIP_B=$SKIP_B"

TEACHER_NUAA="$(pick_teacher NUAA-SIRST)"
TEACHER_NUDT="$(pick_teacher NUDT-SIRST)"
[[ -n "$TEACHER_NUAA" ]] || { echo "[error] missing NUAA teacher"; exit 1; }
[[ -n "$TEACHER_NUDT" || "$SKIP_NUDT" == "1" ]] || { echo "[error] missing NUDT teacher"; exit 1; }

# ----- 1) NUAA: A + B -----
run_group_A "NUAA-SIRST" "$TEACHER_NUAA" "$WORK_ROOT/NUAA/A_distill" "0.7171"
if [[ "$SKIP_B" != "1" ]]; then
  run_group_B_nuaa "$TEACHER_NUAA" "$WORK_ROOT/NUAA/B_criterion" "$WORK_ROOT/NUAA/A_distill/A0_full_pipeline.log"
else
  echo "[skip] Group B criterion"
fi

# ----- 2) NUDT: A only -----
if [[ "$SKIP_NUDT" != "1" ]]; then
  run_group_A "NUDT-SIRST" "$TEACHER_NUDT" "$WORK_ROOT/NUDT/A_distill" "0.8487"
else
  echo "[skip] NUDT Group A"
fi

# ----- global summary -----
SUMMARY="$WORK_ROOT/summary_ablation.txt"
{
  echo "DNANet STABLE ablation summary $(date)"
  echo "plan: NUAA=A+B, NUDT=A_only"
  echo
  echo "==== NUAA Group A ===="
  cat "$WORK_ROOT/NUAA/A_distill/summary_A.txt" 2>/dev/null || true
  echo
  echo "==== NUAA Group B ===="
  cat "$WORK_ROOT/NUAA/B_criterion/summary_B.txt" 2>/dev/null || echo "(skipped or missing)"
  echo
  echo "==== NUDT Group A ===="
  cat "$WORK_ROOT/NUDT/A_distill/summary_A.txt" 2>/dev/null || echo "(skipped or missing)"
} | tee "$SUMMARY"

echo "======== DNA ablation STABLE done $(date) ========"
echo "WORK_ROOT=$WORK_ROOT"
echo "summary -> $SUMMARY"
