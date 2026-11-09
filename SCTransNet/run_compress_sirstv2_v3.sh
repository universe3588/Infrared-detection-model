#!/usr/bin/env bash
# SIRST-V2 再救一档（v3）：更保守 {p_s}，目标相对教师掉点 ≤3 pp
# 教师 best mIoU≈0.7735 → 目标学生 ≳0.7435
# 框架仍为 P1→P2→P3；仅本数据集、仅 SCTrans。
set -euo pipefail
cd "$(dirname "$0")"

export DATASETS="SIRST-V2"
# 强制 v3 配额（避免被 overnight 里更激进的 export 覆盖）
export PRUNE_RATES="0.05,0.15,0.25,0.35"
export EPOCHS="${EPOCHS:-350}"
export LR="${LR:-0.0003}"
export LAMBDA_REGION="${LAMBDA_REGION:-0.75}"
export WORK_DIR="${WORK_DIR:-./compress_runs}"

echo "======== SCTrans SIRST-V2 v3 recovery $(date) ========"
echo "target: IoU drop <= 3pp vs teacher (~0.7735) => student >= ~0.7435"
echo "PRUNE_RATES=$PRUNE_RATES EPOCHS=$EPOCHS LR=$LR"
bash run_compress_all.sh 2>&1 | tee compress_sirstv2_v3.log
echo "======== SIRST-V2 v3 done $(date) ========"
# 跑完后在日志里搜 best_iou= 对照教师
grep -E 'best_iou=|student channel_num|params=' compress_sirstv2_v3.log | tail -20 || true
