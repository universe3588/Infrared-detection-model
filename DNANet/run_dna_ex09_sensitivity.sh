#!/usr/bin/env bash
# Ex09：α 与分层配额 {p_s} 敏感性（精简版，仅 NUAA）
# 说明：每条配置 = 重新 score → prune → distill 200 epoch，比敲除贵很多。
# 用法：
#   cd DNANet
#   sed -i 's/\r$//' run_dna_ex09_sensitivity.sh
#   nohup bash run_dna_ex09_sensitivity.sh > logs/ex09_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs compress_runs

TEACHER="${TEACHER:-result/NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar}"
ROOT="${ROOT:-../RDANet-main/datasets}"
DATASET="NUAA-SIRST"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-8}"
WORK_ROOT="compress_runs/ex09_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_ROOT"

run_one () {
  local name="$1"
  local alpha="$2"
  local rates="$3"
  local wdir="$WORK_ROOT/$name"
  echo "======== [Ex09] $name  alpha=$alpha  rates=$rates ========"
  python compress_train.py --mode pipeline \
    --teacher_ckpt "$TEACHER" \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --epochs "$EPOCHS" \
    --train_batch_size "$BS" \
    --test_batch_size "$BS" \
    --score_batches 50 \
    --deep_supervision True \
    --train_aug resize \
    --lambda_seg 1.0 --lambda_kd 1.0 --lambda_region 0.5 \
    --prune_criterion full \
    --alpha "$alpha" \
    --prune_rates "$rates" \
    --work_dir "$wdir"
}

# ---- A. α 敏感性（配额固定为正文默认）----
DEFAULT_RATES="0.20,0.30,0.40,0.50,0.60"
# 0.0 / 1.0 已在准则消融 contrast/task 中覆盖，这里只补中间点 + 默认
run_one "alpha_0.3" 0.3 "$DEFAULT_RATES"
run_one "alpha_0.5" 0.5 "$DEFAULT_RATES"
run_one "alpha_0.7" 0.7 "$DEFAULT_RATES"   # 与主实验同设定，作对照锚点
run_one "alpha_0.9" 0.9 "$DEFAULT_RATES"

# ---- B. 配额敏感性（α 固定 0.7）----
run_one "quota_mild"   0.7 "0.10,0.20,0.30,0.40,0.50"
run_one "quota_default" 0.7 "0.20,0.30,0.40,0.50,0.60"
run_one "quota_flat"    0.7 "0.40,0.40,0.40,0.40,0.40"
run_one "quota_aggressive" 0.7 "0.30,0.40,0.50,0.60,0.70"

# 汇总 best IoU（从各目录 distill_log / student ckpt 粗提）
SUMMARY="$WORK_ROOT/summary_ex09.csv"
echo "name,alpha,prune_rates,student_best_path" > "$SUMMARY"
find "$WORK_ROOT" -name 'student_best.pth.tar' | sort | while read -r f; do
  name=$(basename "$(dirname "$f")")
  # 上一级常是 pipeline_时间戳
  parent=$(basename "$(dirname "$(dirname "$f")")")
  echo "${parent},${f}" >> "$SUMMARY"
done
echo "[done] Ex09 -> $WORK_ROOT"
echo "[hint] 把各 run 的 best IoU 填进表/曲线即可；正文默认点是 alpha=0.7 + quota_default"
