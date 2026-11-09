#!/usr/bin/env bash
# 补跑消融失败项（上一轮核实）：
#   1) NUDT A4：完整配方 distill（优先复用已有 keep/init；并加一份 λ_kd=0.5 保底）
#   2) NUAA ℓ1：准则 pipeline 整段重跑
#
# 用法（在 Infrared-Small-Target-Detection-master 下）：
#   sed -i 's/\r$//' run_dna_ablation_repair.sh
#   nohup bash run_dna_ablation_repair.sh > logs/ablation_repair.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p logs

ROOT_DATA="${ROOT_DATA:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-8}"
PRUNE_RATES="${PRUNE_RATES:-0.20,0.30,0.40,0.50,0.60}"

# 默认指向上一轮 stable 目录；若服务器路径不同可：
#   ABLATION_ROOT=./compress_runs/ablation_stable_XXXX bash run_dna_ablation_repair.sh
ABLATION_ROOT="${ABLATION_ROOT:-./compress_runs/ablation_stable_20261108_112424}"
REPAIR_ROOT="${REPAIR_ROOT:-./compress_runs/ablation_repair_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$REPAIR_ROOT"

pick_teacher() {
  local ds="$1"
  ls -1td result/${ds}_DNANet_*_wDS/mIoU__DNANet_*_epoch.pth.tar 2>/dev/null | head -1 || true
}

echo "======== DNA ablation REPAIR start $(date) ========"
echo "ABLATION_ROOT=$ABLATION_ROOT"
echo "REPAIR_ROOT=$REPAIR_ROOT"

TEACHER_NUAA="$(pick_teacher NUAA-SIRST)"
TEACHER_NUDT="$(pick_teacher NUDT-SIRST)"
[[ -n "$TEACHER_NUAA" ]] || { echo "[error] missing NUAA teacher"; exit 1; }
[[ -n "$TEACHER_NUDT" ]] || { echo "[error] missing NUDT teacher"; exit 1; }

# ---------------------------------------------------------------------------
# 1) NUDT A4：复用 keep + student_init
# ---------------------------------------------------------------------------
NUDT_A="$ABLATION_ROOT/NUDT/A_distill"
if [[ -f "$NUDT_A/BASE_PIPE.txt" ]]; then
  # BASE_PIPE=./compress_runs/...
  # shellcheck disable=SC1090
  source "$NUDT_A/BASE_PIPE.txt"
  NUDT_PIPE="${BASE_PIPE}"
else
  NUDT_PIPE="$(ls -1td "$NUDT_A"/NUDT-SIRST_pipeline_* 2>/dev/null | head -1 || true)"
fi

KEEP="${NUDT_PIPE:-}/keep_masks.pt"
INIT="${NUDT_PIPE:-}/student_init.pth.tar"

if [[ ! -f "$KEEP" || ! -f "$INIT" ]]; then
  echo "[warn] NUDT keep/init missing under $ABLATION_ROOT ; will score+prune once"
  PRE="$REPAIR_ROOT/NUDT_prepare"
  mkdir -p "$PRE"
  python -u compress_train.py --mode score \
    --teacher_ckpt "$TEACHER_NUDT" \
    --dataset NUDT-SIRST --root "$ROOT_DATA" \
    --deep_supervision True --train_aug resize \
    --train_batch_size "$BS" --test_batch_size "$BS" \
    --prune_rates "$PRUNE_RATES" --prune_criterion full --alpha 0.7 \
    --seed 42 --workers 4 \
    --work_dir "$PRE" 2>&1 | tee "$PRE/score.log"
  SCORE_DIR="$(ls -1td "$PRE"/NUDT-SIRST_score_* 2>/dev/null | head -1)"
  KEEP="$SCORE_DIR/keep_masks.pt"
  python -u compress_train.py --mode prune \
    --teacher_ckpt "$TEACHER_NUDT" \
    --dataset NUDT-SIRST --root "$ROOT_DATA" \
    --deep_supervision True --train_aug resize \
    --keep_path "$KEEP" \
    --work_dir "$PRE" 2>&1 | tee "$PRE/prune.log"
  PRUNE_DIR="$(ls -1td "$PRE"/NUDT-SIRST_prune_* 2>/dev/null | head -1)"
  INIT="$PRUNE_DIR/student_init.pth.tar"
  # prune 也写 keep 到 work 子目录时，统一用 SCORE keep
fi

echo "[NUDT] KEEP=$KEEP"
echo "[NUDT] INIT=$INIT"
[[ -f "$KEEP" && -f "$INIT" ]] || { echo "[error] still no KEEP/INIT"; exit 1; }

NUDT_COMMON=(
  --teacher_ckpt "$TEACHER_NUDT"
  --dataset NUDT-SIRST
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
  --keep_path "$KEEP"
  --student_ckpt "$INIT"
)

echo "---- [NUDT][A4] distill full λ_kd=1.0 ----"
mkdir -p "$REPAIR_ROOT/NUDT_A4_kd1"
python -u compress_train.py --mode distill \
  "${NUDT_COMMON[@]}" \
  --lambda_seg 1.0 --lambda_region 0.5 --lambda_kd 1.0 \
  --work_dir "$REPAIR_ROOT/NUDT_A4_kd1" \
  2>&1 | tee "$REPAIR_ROOT/NUDT_A4_kd1/train.log"

echo "---- [NUDT][A4b] distill λ_kd=0.5（保底，防再塌） ----"
mkdir -p "$REPAIR_ROOT/NUDT_A4_kd05"
python -u compress_train.py --mode distill \
  "${NUDT_COMMON[@]}" \
  --lambda_seg 1.0 --lambda_region 0.5 --lambda_kd 0.5 \
  --work_dir "$REPAIR_ROOT/NUDT_A4_kd05" \
  2>&1 | tee "$REPAIR_ROOT/NUDT_A4_kd05/train.log"

# ---------------------------------------------------------------------------
# 2) NUAA ℓ1 准则整段重跑
# ---------------------------------------------------------------------------
echo "---- [NUAA][B-l1] pipeline prune_criterion=l1 ----"
mkdir -p "$REPAIR_ROOT/NUAA_l1"
python -u compress_train.py --mode pipeline \
  --teacher_ckpt "$TEACHER_NUAA" \
  --dataset NUAA-SIRST --root "$ROOT_DATA" \
  --deep_supervision True --train_aug resize \
  --train_batch_size "$BS" --test_batch_size "$BS" \
  --prune_rates "$PRUNE_RATES" --prune_criterion l1 --alpha 0.7 \
  --epochs "$EPOCHS" --lr 0.05 --optimizer Adagrad \
  --lambda_seg 1.0 --lambda_region 0.5 --lambda_kd 1.0 \
  --seed 42 --workers 4 \
  --work_dir "$REPAIR_ROOT/NUAA_l1" \
  2>&1 | tee "$REPAIR_ROOT/NUAA_l1/train.log"

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
SUMMARY="$REPAIR_ROOT/summary_repair.txt"
{
  echo "DNA ablation REPAIR summary $(date)"
  echo "ABLATION_ROOT=$ABLATION_ROOT"
  echo "REPAIR_ROOT=$REPAIR_ROOT"
  echo
  echo -n "NUDT A4 kd=1.0: "; grep -E 'best mIoU=|best student mIoU=' "$REPAIR_ROOT/NUDT_A4_kd1/train.log" 2>/dev/null | tail -1 || true
  echo -n "NUDT A4b kd=0.5: "; grep -E 'best mIoU=|best student mIoU=' "$REPAIR_ROOT/NUDT_A4_kd05/train.log" 2>/dev/null | tail -1 || true
  echo -n "NUAA l1: "; grep -E 'best mIoU=|best student mIoU=' "$REPAIR_ROOT/NUAA_l1/train.log" 2>/dev/null | tail -1 || true
  echo
  echo "Prev valid refs:"
  echo "  NUAA A2=0.7454 A3=0.7551 A4(old)=0.7188 teacher=0.7171"
  echo "  NUDT A2=0.9149 A3=0.9180 teacher=0.8487"
} | tee "$SUMMARY"

echo "======== DNA ablation REPAIR done $(date) ========"
echo "summary -> $SUMMARY"
echo "拉回整个目录: $REPAIR_ROOT"
