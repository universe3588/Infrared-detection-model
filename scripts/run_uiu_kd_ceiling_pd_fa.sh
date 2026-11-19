#!/usr/bin/env bash
# =============================================================================
# 评测 UIU-Net 强教师 KD 消融 A3 / A4 的统一脚本 P_d / F_a（NUAA-SIRST）
#
# 用法（项目根目录）：
#   cd ~/autodl-tmp/Infrared-detection-model
#   bash scripts/run_uiu_kd_ceiling_pd_fa.sh
#
# 可选：
#   KD_DIR=UIUNet/compress_runs/kd_ceiling_NUAA-SIRST_20261119_211657 \
#     bash scripts/run_uiu_kd_ceiling_pd_fa.sh
#   WITH_TEACHER=0   # 只评学生，更快（默认 1：教师+学生对照）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

GPUS="${GPUS:-0}"
WITH_TEACHER="${WITH_TEACHER:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/pd_fa_eval/uiu_kd_ceiling}"
DATASET_DIR="${DATASET_DIR:-$ROOT/RDANet-main/datasets}"
mkdir -p "$OUT_ROOT" logs

echo "======== UIU KD-ceiling P_d/F_a $(date) ========"
echo "ROOT=$ROOT"

# 评测脚本内部 repo 名为 IEEE_TIP_UIU-Net-main；服务器目录常为 UIUNet
if [[ ! -d "$ROOT/IEEE_TIP_UIU-Net-main" ]]; then
  if [[ -d "$ROOT/UIUNet" ]]; then
    ln -sfn UIUNet "$ROOT/IEEE_TIP_UIU-Net-main"
    echo "[link] IEEE_TIP_UIU-Net-main -> UIUNet"
  else
    echo "[err] 找不到 UIUNet 或 IEEE_TIP_UIU-Net-main"
    exit 1
  fi
fi

# 定位消融目录
if [[ -n "${KD_DIR:-}" ]]; then
  KD="$KD_DIR"
  [[ "$KD" = /* ]] || KD="$ROOT/$KD"
else
  KD="$(ls -1td "$ROOT"/UIUNet/compress_runs/kd_ceiling_NUAA-SIRST_* 2>/dev/null | head -1 || true)"
  if [[ -z "$KD" ]]; then
    KD="$(ls -1td "$ROOT"/IEEE_TIP_UIU-Net-main/compress_runs/kd_ceiling_NUAA-SIRST_* 2>/dev/null | head -1 || true)"
  fi
fi
[[ -n "$KD" && -d "$KD" ]] || { echo "[err] 找不到 kd_ceiling_NUAA-SIRST_* 目录，请设 KD_DIR=..."; exit 1; }
echo "KD_DIR=$KD"

A3="$(ls -1t "$KD"/A3_seg_region/NUAA-SIRST_distill_*/student_best.pth.tar 2>/dev/null | head -1 || true)"
A4="$(ls -1t "$KD"/A4_seg_region_kd1/NUAA-SIRST_distill_*/student_best.pth.tar 2>/dev/null | head -1 || true)"
[[ -f "$A3" ]] || { echo "[err] 缺 A3 student_best: $KD/A3_seg_region/..."; exit 1; }
[[ -f "$A4" ]] || { echo "[err] 缺 A4 student_best: $KD/A4_seg_region_kd1/..."; exit 1; }
echo "A3=$A3"
echo "A4=$A4"

TCH_CAND=(
  "$ROOT/UIUNet/saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar"
  "$ROOT/IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar"
)
TCH=""
for t in "${TCH_CAND[@]}"; do
  if [[ -f "$t" ]]; then TCH="$t"; break; fi
done
[[ -n "$TCH" ]] || { echo "[err] 缺 UIU 教师权重 uiunet_best.pth.tar"; exit 1; }
echo "TEACHER=$TCH"

# 路径改成相对 ROOT，并把 UIUNet/ 换成 IEEE_TIP_UIU-Net-main/
rel_repo() {
  local p="$1"
  p="$(realpath "$p")"
  p="${p#$ROOT/}"
  p="${p/UIUNet\//IEEE_TIP_UIU-Net-main/}"
  echo "$p"
}

TCH_REL="$(rel_repo "$TCH")"
A3_REL="$(rel_repo "$A3")"
A4_REL="$(rel_repo "$A4")"

if [[ "$WITH_TEACHER" == "1" ]]; then
  ROLES="teacher,student"
else
  ROLES="student"
fi

write_manifest() {
  local name="$1" stu_rel="$2" out="$3"
  python - "$out" "$TCH_REL" "$stu_rel" "$WITH_TEACHER" <<'PY'
import json, sys
out, tch, stu, with_t = sys.argv[1:5]
jobs = []
if with_t == "1":
    jobs.append({"backbone": "uiu", "dataset": "NUAA-SIRST", "role": "teacher", "ckpt": tch})
jobs.append({"backbone": "uiu", "dataset": "NUAA-SIRST", "role": "student", "ckpt": stu})
with open(out, "w", encoding="utf-8") as f:
    json.dump({"jobs": jobs}, f, indent=2, ensure_ascii=False)
print("[manifest]", out)
PY
}

LOG="$OUT_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
{
  echo "KD=$KD"
  echo "A3=$A3"
  echo "A4=$A4"
  echo "TCH=$TCH"
  echo "ROLES=$ROLES"
} | tee "$OUT_ROOT/RUN_META.txt"

for NAME in A3 A4; do
  if [[ "$NAME" == "A3" ]]; then STU_REL="$A3_REL"; else STU_REL="$A4_REL"; fi
  MAN="$OUT_ROOT/manifest_${NAME}.json"
  OUTD="$OUT_ROOT/$NAME"
  mkdir -p "$OUTD"
  write_manifest "$NAME" "$STU_REL" "$MAN"

  echo ""
  echo "=============================================================================="
  echo "[评测] $NAME  roles=$ROLES"
  echo "=============================================================================="
  python -u scripts/eval_pd_fa_unified.py \
    --root "$ROOT" \
    --dataset_dir "$DATASET_DIR" \
    --backbone uiu \
    --datasets NUAA-SIRST \
    --roles "$ROLES" \
    --manifest "$MAN" \
    --out_dir "$OUTD" \
    --gpus "$GPUS" \
    2>&1 | tee -a "$LOG"

  echo "---- $NAME latest ----"
  ls -lh "$OUTD"/pd_fa_*_latest.* 2>/dev/null || true
  cat "$OUTD"/pd_fa_*_latest.md 2>/dev/null || true
done

# 对照摘要
{
  echo "======== UIU KD-ceiling P_d/F_a summary $(date) ========"
  echo "KD=$KD"
  echo ""
  echo "---- A3 (seg+region, λ_kd=0) ----"
  cat "$OUT_ROOT/A3"/pd_fa_*_latest.md 2>/dev/null || echo "(missing)"
  echo ""
  echo "---- A4 (seg+region+KD, λ_kd=1) ----"
  cat "$OUT_ROOT/A4"/pd_fa_*_latest.md 2>/dev/null || echo "(missing)"
} | tee "$OUT_ROOT/SUMMARY.md"

echo ""
echo "[done] 摘要: $OUT_ROOT/SUMMARY.md"
echo "       日志: $LOG"
