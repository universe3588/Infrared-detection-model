#!/usr/bin/env bash
# =============================================================================
# UIU-Net 强教师 KD 消融（Teacher Ceiling 闭环）
#
# 用法：
#   cd ~/autodl-tmp/Infrared-detection-model/UIUNet
#   mkdir -p logs
#   # 缺 keep_masks/student_init 时：
#   BOOTSTRAP=1 bash run_uiu_kd_ceiling_ablation.sh 2>&1 | tee logs/uiu_kd_ceiling_nuaa.log
#   # 已有 masks：
#   BASE_PIPE=./compress_runs/xxx bash run_uiu_kd_ceiling_ablation.sh 2>&1 | tee logs/uiu_kd_ceiling_nuaa.log
#
# 环境变量：DATASET / EPOCHS / BS / RUN_KD05 / BOOTSTRAP / BASE_PIPE / TEACHER
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p logs compress_runs

DATASET="${DATASET:-NUAA-SIRST}"
DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-3}"
GPUS="${GPUS:-0}"
SEED="${SEED:-42}"
RUN_KD05="${RUN_KD05:-0}"
BOOTSTRAP="${BOOTSTRAP:-0}"
WORK_ROOT="${WORK_ROOT:-./compress_runs/kd_ceiling_${DATASET}_$(date +%Y%m%d_%H%M%S)}"

declare -A DEFAULT_TEACHER=(
  ["NUAA-SIRST"]="saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar"
  ["SIRST-V2"]="saved_models/uiunet_irstd/SIRST-V2_20261030_224844/uiunet_best.pth.tar"
)
TEACHER="${TEACHER:-${DEFAULT_TEACHER[$DATASET]}}"

# -------------------- 进度工具 --------------------
ALL_T0=$(date +%s)
STEP_T0=$ALL_T0
STEP_I=0
STEP_N=0
STEP_NAME=""

fmt_hms() {
  local sec=$1
  if (( sec < 0 )); then sec=0; fi
  printf "%02d:%02d:%02d" $((sec/3600)) $(((sec%3600)/60)) $((sec%60))
}

progress_banner() {
  local title="$1"
  local detail="${2:-}"
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - ALL_T0))
  echo ""
  echo "=============================================================================="
  echo "[进度] ${STEP_I}/${STEP_N}  ${title}"
  if [[ -n "$detail" ]]; then
    echo "       ${detail}"
  fi
  echo "       已用时=$(fmt_hms "$elapsed")  |  $(date '+%F %T')"
  echo "=============================================================================="
}

begin_step() {
  STEP_I=$((STEP_I + 1))
  STEP_NAME="$1"
  STEP_T0=$(date +%s)
  progress_banner "$STEP_NAME" "${2:-}"
}

end_step() {
  local now dt
  now=$(date +%s)
  dt=$((now - STEP_T0))
  echo ""
  echo "[完成] ${STEP_I}/${STEP_N}  ${STEP_NAME}  本步耗时=$(fmt_hms "$dt")"
  # 粗估剩余：按已完成步均时 × 剩余步（蒸馏步更重，仅作量级参考）
  if (( STEP_I > 0 && STEP_I < STEP_N )); then
    local avg remain
    avg=$(( (now - ALL_T0) / STEP_I ))
    remain=$(( avg * (STEP_N - STEP_I) ))
    echo "[预估] 剩余约 $(fmt_hms "$remain")（按已完成步均时粗估，蒸馏步可能更久）"
  fi
}

# -------------------- 检查教师 --------------------
if [[ ! -f "$TEACHER" ]]; then
  echo "[err] 缺少教师权重: $TEACHER"
  echo "      请上传本地："
  echo "      UIUNet/saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar"
  exit 1
fi

# -------------------- 解析是否需要 bootstrap --------------------
if [[ -z "${BASE_PIPE:-}" ]]; then
  BASE_PIPE="$(ls -1td ./compress_runs/${DATASET}_pipeline_* 2>/dev/null | head -1 || true)"
fi

NEED_BOOTSTRAP=0
if [[ -z "${BASE_PIPE:-}" || ! -d "${BASE_PIPE:-}" ]]; then
  NEED_BOOTSTRAP=1
elif [[ ! -f "$BASE_PIPE/keep_masks.pt" || ! -f "$BASE_PIPE/student_init.pth.tar" ]]; then
  echo "[warn] pipeline 目录在，但缺 keep_masks/student_init: $BASE_PIPE"
  NEED_BOOTSTRAP=1
fi

if [[ "$NEED_BOOTSTRAP" == "1" && "$BOOTSTRAP" != "1" ]]; then
  echo "[err] 找不到可用的 ${DATASET} keep_masks + student_init。"
  echo "      (1) 从旧实例拷 keep_masks.pt + student_init.pth.tar 后设 BASE_PIPE=目录"
  echo "      (2) 或：BOOTSTRAP=1 bash run_uiu_kd_ceiling_ablation.sh"
  exit 1
fi

# -------------------- 规划总步数（终端一开始就能看到路线图）--------------------
STEPS=()
if [[ "$NEED_BOOTSTRAP" == "1" ]]; then
  STEPS+=("BOOTSTRAP-score：通道敏感性评分")
  STEPS+=("BOOTSTRAP-prune：结构化剪枝生成 student_init")
fi
STEPS+=("A3：分割+区域（λ_kd=0）蒸馏 ${EPOCHS} epoch")
STEPS+=("A4：分割+区域+KD（λ_kd=1）蒸馏 ${EPOCHS} epoch")
if [[ "$RUN_KD05" == "1" ]]; then
  STEPS+=("A4b：分割+区域+KD（λ_kd=0.5）蒸馏 ${EPOCHS} epoch")
fi
STEPS+=("汇总：写 SUMMARY.txt")
STEP_N=${#STEPS[@]}

echo ""
echo "=============================================================================="
echo " UIU-Net 强教师 KD 消融   dataset=${DATASET}"
echo " 总步骤=${STEP_N}   epochs=${EPOCHS}   bs=${BS}   seed=${SEED}"
echo " teacher=${TEACHER}"
echo " BOOTSTRAP=${BOOTSTRAP}  RUN_KD05=${RUN_KD05}"
echo " work_root=${WORK_ROOT}"
echo "------------------------------------------------------------------------------"
echo " 路线图："
local_i=0
for s in "${STEPS[@]}"; do
  local_i=$((local_i + 1))
  printf "   %d/%d  %s\n" "$local_i" "$STEP_N" "$s"
done
echo "=============================================================================="
echo ""

# -------------------- bootstrap：score + prune --------------------
if [[ "$NEED_BOOTSTRAP" == "1" ]]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  BOOT_DIR="./compress_runs/${DATASET}_bootstrap_${STAMP}"
  mkdir -p "$BOOT_DIR"

  begin_step "BOOTSTRAP-score：通道敏感性评分" "输出 keep_masks.pt → ${BOOT_DIR}"
  python -u compress_train.py --mode score \
    --teacher_ckpt "$TEACHER" \
    --dataset "$DATASET" --dataset_dir "$DATASET_DIR" \
    --batch_size "$BS" --rescale 320 --crop 288 \
    --prune_rates 0.20,0.30,0.40,0.50 \
    --gpus "$GPUS" --seed "$SEED" \
    --work_dir "$BOOT_DIR" 2>&1 | tee "logs/uiu_bootstrap_${DATASET}.log"
  SCORE_DIR="$(ls -1td "$BOOT_DIR"/${DATASET}_score_* 2>/dev/null | head -1)"
  KEEP0="$SCORE_DIR/keep_masks.pt"
  [[ -f "$KEEP0" ]] || { echo "[err] score 未产出 keep_masks: $KEEP0"; exit 1; }
  end_step

  begin_step "BOOTSTRAP-prune：结构化剪枝" "keep=$KEEP0"
  python -u compress_train.py --mode prune \
    --teacher_ckpt "$TEACHER" \
    --keep_path "$KEEP0" \
    --dataset "$DATASET" --dataset_dir "$DATASET_DIR" \
    --batch_size "$BS" --rescale 320 --crop 288 \
    --gpus "$GPUS" --seed "$SEED" \
    --work_dir "$BOOT_DIR" 2>&1 | tee -a "logs/uiu_bootstrap_${DATASET}.log"
  PRUNE_DIR="$(ls -1td "$BOOT_DIR"/${DATASET}_prune_* 2>/dev/null | head -1)"
  INIT0="$PRUNE_DIR/student_init.pth.tar"
  [[ -f "$INIT0" ]] || { echo "[err] prune 未产出 student_init: $INIT0"; exit 1; }
  BASE_PIPE="$BOOT_DIR/shared_masks"
  mkdir -p "$BASE_PIPE"
  cp -f "$KEEP0" "$BASE_PIPE/keep_masks.pt"
  cp -f "$INIT0" "$BASE_PIPE/student_init.pth.tar"
  echo "[bootstrap] BASE_PIPE=$BASE_PIPE"
  end_step
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
  echo "total_steps=$STEP_N"
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
  local name="$1" kd="$2" title="$3"
  local out="$WORK_ROOT/$name"
  mkdir -p "$out"
  begin_step "$title" "λ_kd=${kd}  epochs=${EPOCHS}  log=${out}/train.log"
  echo "[提示] 下面是 Python 训练日志；本步结束后会再打印整体进度条。"
  python -u compress_train.py \
    "${COMMON[@]}" \
    --lambda_kd "$kd" \
    --work_dir "$out" \
    2>&1 | tee "$out/train.log"
  local run_dir
  run_dir="$(ls -1td "$out"/${DATASET}_distill_* 2>/dev/null | head -1 || true)"
  if [[ -n "$run_dir" ]]; then
    echo "$run_dir" > "$out/ACTUAL_RUN_DIR.txt"
    grep -E 'best student IoU=|best mIoU=|teacher eval|test_IoU=' "$out/train.log" | tail -20 \
      | tee "$out/best_snip.txt" || true
  fi
  # 本步结果速览
  echo -n "[本步结果] ${name}: "
  grep -E 'best student IoU=' "$out/train.log" 2>/dev/null | tail -1 || echo "(见 train.log)"
  end_step
}

# A3 / A4 / 可选 A4b
run_one "A3_seg_region" 0.0 "A3：分割+区域（λ_kd=0）蒸馏"
run_one "A4_seg_region_kd1" 1.0 "A4：分割+区域+KD（λ_kd=1）蒸馏"
if [[ "$RUN_KD05" == "1" ]]; then
  run_one "A4b_seg_region_kd05" 0.5 "A4b：分割+区域+KD（λ_kd=0.5）蒸馏"
fi

# -------------------- 汇总 --------------------
begin_step "汇总：写 SUMMARY.txt" "目录=$WORK_ROOT"
{
  echo "======== UIU KD ceiling summary $(date) ========"
  echo "teacher=$TEACHER"
  echo "base_pipe=$BASE_PIPE"
  echo "主表参照：UIU 教师 NUAA=0.7686 / 学生完整配方=0.7741"
  echo "总耗时=$(fmt_hms $(( $(date +%s) - ALL_T0 )))"
  echo ""
  for d in A3_seg_region A4_seg_region_kd1 A4b_seg_region_kd05; do
    [[ -d "$WORK_ROOT/$d" ]] || continue
    echo -n "$d: "
    if [[ -f "$WORK_ROOT/$d/best_snip.txt" ]]; then
      grep -E 'best student IoU=' "$WORK_ROOT/$d/best_snip.txt" | tail -1 \
        || tail -3 "$WORK_ROOT/$d/best_snip.txt"
    else
      grep -E 'best student IoU=' "$WORK_ROOT/$d/train.log" 2>/dev/null | tail -1 \
        || echo "(no best line yet)"
    fi
  done
  echo ""
  echo "判读："
  echo "  Δ = A4(kd=1) - A3(no kd)"
  echo "  Δ ≥ +0.005  → 强教师上 KD 有正向增益，Teacher Ceiling 叙事闭环"
  echo "  |Δ| < 0.003 → KD 近似中性，正文继续写『可选正则/校准项』"
  echo "  Δ ≤ -0.005  → 即使强教师也偏负；保留 KD 但默认权重要下调叙事"
} | tee "$WORK_ROOT/SUMMARY.txt"
end_step

echo ""
echo "=============================================================================="
echo "[全部完成] ${STEP_N}/${STEP_N}  总耗时=$(fmt_hms $(( $(date +%s) - ALL_T0 )))"
echo "  结果目录: $WORK_ROOT"
echo "  汇总文件: $WORK_ROOT/SUMMARY.txt"
echo "=============================================================================="
