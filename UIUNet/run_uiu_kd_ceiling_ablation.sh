#!/usr/bin/env bash
# =============================================================================
# UIU-Net 强教师 KD 消融（Teacher Ceiling 闭环）
#
# 目的：在强教师（UIU-Net/NUAA，教师 IoU≈0.7686）上检验
#       「分割+区域」 vs 「分割+区域+输出KD」是否带来正向增益。
#
# 关键：复用主实验同一 keep_masks + student_init，只改蒸馏损失权重。
#
# 用法（AutoDL / 服务器）：
#   cd /root/autodl-tmp/IEEE_TIP_UIU-Net-main   # 或你的 UIUNet 目录
#   bash run_uiu_kd_ceiling_ablation.sh
#
# 可选环境变量：
#   DATASET=NUAA-SIRST|SIRST-V2   默认 NUAA-SIRST
#   EPOCHS=200
#   BS=3
#   RUN_KD05=1                    额外跑 λ_kd=0.5（默认 0，半天量级可不跑）
#   BASE_PIPE=...                 指定已有 pipeline 目录（含 keep_masks/student_init）
#   TEACHER=...                   教师权重路径
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

DATASET="${DATASET:-NUAA-SIRST}"
DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-3}"
GPUS="${GPUS:-0}"
SEED="${SEED:-42}"
RUN_KD05="${RUN_KD05:-0}"
WORK_ROOT="${WORK_ROOT:-./compress_runs/kd_ceiling_${DATASET}_$(date +%Y%m%d_%H%M%S)}"

# ---- 教师（与主实验一致）----
declare -A DEFAULT_TEACHER=(
  ["NUAA-SIRST"]="saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar"
  ["SIRST-V2"]="saved_models/uiunet_irstd/SIRST-V2_20261030_224844/uiunet_best.pth.tar"
)
TEACHER="${TEACHER:-${DEFAULT_TEACHER[$DATASET]}}"

# ---- 复用主实验剪枝产物（同一 keep_masks，才是干净对照）----
if [[ -z "${BASE_PIPE:-}" ]]; then
  BASE_PIPE="$(ls -1td ./compress_runs/${DATASET}_pipeline_* 2>/dev/null | head -1 || true)"
fi
if [[ -z "$BASE_PIPE" || ! -d "$BASE_PIPE" ]]; then
  echo "[err] 找不到 ${DATASET} 的主实验 pipeline 目录。"
  echo "      请先设 BASE_PIPE=/path/to/NUAA-SIRST_pipeline_xxx"
  echo "      或先跑一遍主 pipeline 再消融。"
  exit 1
fi
KEEP="$BASE_PIPE/keep_masks.pt"
INIT="$BASE_PIPE/student_init.pth.tar"
for f in "$TEACHER" "$KEEP" "$INIT"; do
  [[ -f "$f" ]] || { echo "[err] missing: $f"; exit 1; }
done

mkdir -p "$WORK_ROOT"
{
  echo "dataset=$DATASET"
  echo "teacher=$TEACHER"
  echo "base_pipe=$BASE_PIPE"
  echo "keep=$KEEP"
  echo "init=$INIT"
  echo "epochs=$EPOCHS bs=$BS seed=$SEED"
  echo "started=$(date)"
} | tee "$WORK_ROOT/RUN_META.txt"

COMMON=(
  --mode distill
  --teacher_ckpt "$TEACHER"
  --keep_path "$KEEP"
  --student_ckpt "$INIT"
  --dataset "$DATASET"
  --dataset_dir "$DATASET_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BS"
  --rescale 320
  --crop 288
  --lr 1e-3
  --optimizer Adam
  --lambda_seg 1.0
  --lambda_region 0.5
  --gpus "$GPUS"
  --seed "$SEED"
)

run_one() {
  local name="$1" kd="$2"
  local out="$WORK_ROOT/$name"
  mkdir -p "$out"
  echo ""
  echo "======== [$name] λ_kd=$kd | $(date) ========"
  python -u compress_train.py \
    "${COMMON[@]}" \
    --lambda_kd "$kd" \
    --work_dir "$out" \
    2>&1 | tee "$out/train.log"
  # compress_train 会在 work_dir 下再建 <DS>_distill_<stamp>
  local run_dir
  run_dir="$(ls -1td "$out"/${DATASET}_distill_* 2>/dev/null | head -1 || true)"
  if [[ -n "$run_dir" ]]; then
    echo "$run_dir" > "$out/ACTUAL_RUN_DIR.txt"
    grep -E 'best student IoU=|best mIoU=|teacher eval|test_IoU=' "$out/train.log" | tail -20 \
      | tee "$out/best_snip.txt" || true
  fi
}

# A3：分割 + 区域（无 KD）
run_one "A3_seg_region" 0.0

# A4：分割 + 区域 + 输出 KD（λ=1）—— 主对照
run_one "A4_seg_region_kd1" 1.0

# 可选：λ_kd=0.5（若 A4 崩溃/掉点再开）
if [[ "$RUN_KD05" == "1" ]]; then
  run_one "A4b_seg_region_kd05" 0.5
fi

# ---- 汇总 ----
{
  echo "======== UIU KD ceiling summary $(date) ========"
  echo "teacher=$TEACHER"
  echo "base_pipe=$BASE_PIPE"
  echo "主表参照：UIU 教师 NUAA=0.7686 / 学生完整配方=0.7741"
  echo ""
  for d in A3_seg_region A4_seg_region_kd1 A4b_seg_region_kd05; do
    [[ -d "$WORK_ROOT/$d" ]] || continue
    echo -n "$d: "
    if [[ -f "$WORK_ROOT/$d/best_snip.txt" ]]; then
      grep -E 'best student IoU=' "$WORK_ROOT/$d/best_snip.txt" | tail -1 || cat "$WORK_ROOT/$d/best_snip.txt" | tail -3
    else
      grep -E 'best student IoU=' "$WORK_ROOT/$d/train.log" 2>/dev/null | tail -1 || echo "(no best line yet)"
    fi
  done
  echo ""
  echo "判读："
  echo "  Δ = A4(kd=1) - A3(no kd)"
  echo "  Δ ≥ +0.005  → 强教师上 KD 有正向增益，Teacher Ceiling 叙事闭环"
  echo "  |Δ| < 0.003 → KD 近似中性，正文继续写『可选正则/校准项』"
  echo "  Δ ≤ -0.005  → 即使强教师也偏负；保留 KD 但默认权重要下调叙事"
} | tee "$WORK_ROOT/SUMMARY.txt"

echo ""
echo "[done] results in $WORK_ROOT"
echo "把 SUMMARY.txt + 两个 student_best 的 best IoU 回传即可填表。"
