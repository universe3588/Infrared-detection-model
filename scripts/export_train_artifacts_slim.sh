#!/usr/bin/env bash
# 在 AutoDL 仓库根目录旁运行：导出可验收的精简训练产物并强压缩打包
# 用法：
#   bash scripts/export_train_artifacts_slim.sh
#   bash scripts/export_train_artifacts_slim.sh /root/autodl-tmp
#
# 包含：
#   - MSHNet：各 run 全量 weight/checkpoint/metric.log
#   - UIUNet：每数据集仅 uiunet_best + args/train_log + logs/
#   - SCTransNet：训练 txt、otherlogs、每数据集 SCTransNet_1000 + 最新 *_best
# 不包含：UIU 中间 epoch_*.pth.tar、SCTrans 历史多余 best（体积大且验收不需要）
# 注意：产物勿 git add / push，仅本机或网盘下载

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${1:-/root/autodl-tmp}"
EXPORT_DIR="${OUT_BASE}/export_slim"
STAMP="$(date +%Y%m%d_%H%M%S)"
TAR_GZ="${OUT_BASE}/train_artifacts_slim_${STAMP}.tar.gz"
MANIFEST="${EXPORT_DIR}/MANIFEST.txt"

echo "======== export slim start $(date) ========"
echo "ROOT=$ROOT"
echo "EXPORT_DIR=$EXPORT_DIR"
echo "TAR_GZ=$TAR_GZ"

rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

copy_tree() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "[SKIP] missing: $src"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "[OK] $src -> $dst"
}

# ----- MSHNet -----
echo "----- MSHNet -----"
if [[ -d "$ROOT/MSHNet-main/weight" ]]; then
  mkdir -p "$EXPORT_DIR/MSHNet-main"
  cp -a "$ROOT/MSHNet-main/weight" "$EXPORT_DIR/MSHNet-main/"
  echo "[OK] MSHNet-main/weight"
else
  echo "[SKIP] MSHNet-main/weight missing"
fi

# ----- UIUNet -----
echo "----- UIUNet -----"
UIU_SRC="$ROOT/IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd"
UIU_DST="$EXPORT_DIR/IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd"
mkdir -p "$UIU_DST"

if [[ -d "$UIU_SRC" ]]; then
  # 优先正式四数据集 run；若命名变化则回退为「每数据集取最新含 best 的目录」
  PREFERRED=(
    NUAA-SIRST_20261030_151749
    NUDT-SIRST_20261030_164112
    IRSTD-1k_20261030_193208
    SIRST-V2_20261030_224844
  )
  copied_any=0
  for d in "${PREFERRED[@]}"; do
    src="$UIU_SRC/$d"
    if [[ -d "$src" && -f "$src/uiunet_best.pth.tar" ]]; then
      dst="$UIU_DST/$d"
      mkdir -p "$dst"
      cp -a "$src/uiunet_best.pth.tar" "$dst/"
      [[ -f "$src/args.txt" ]] && cp -a "$src/args.txt" "$dst/"
      [[ -f "$src/train_log.csv" ]] && cp -a "$src/train_log.csv" "$dst/"
      [[ -f "$src/metric.log" ]] && cp -a "$src/metric.log" "$dst/"
      echo "[OK] UIU slim $d"
      copied_any=1
    fi
  done

  if [[ "$copied_any" -eq 0 ]]; then
    echo "[WARN] preferred UIU dirs not found; auto-pick latest best per dataset prefix"
    for prefix in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
      best_dir=""
      # 按修改时间取最新含 uiunet_best 的目录
      while IFS= read -r cand; do
        [[ -f "$cand/uiunet_best.pth.tar" ]] || continue
        best_dir="$cand"
        break
      done < <(find "$UIU_SRC" -maxdepth 1 -type d -name "${prefix}_*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}')
      if [[ -n "$best_dir" ]]; then
        name="$(basename "$best_dir")"
        dst="$UIU_DST/$name"
        mkdir -p "$dst"
        cp -a "$best_dir/uiunet_best.pth.tar" "$dst/"
        [[ -f "$best_dir/args.txt" ]] && cp -a "$best_dir/args.txt" "$dst/"
        [[ -f "$best_dir/train_log.csv" ]] && cp -a "$best_dir/train_log.csv" "$dst/"
        [[ -f "$best_dir/metric.log" ]] && cp -a "$best_dir/metric.log" "$dst/"
        echo "[OK] UIU auto $name"
      else
        echo "[SKIP] UIU no best for $prefix"
      fi
    done
  fi
else
  echo "[SKIP] $UIU_SRC missing"
fi

if [[ -d "$ROOT/IEEE_TIP_UIU-Net-main/logs" ]]; then
  mkdir -p "$EXPORT_DIR/IEEE_TIP_UIU-Net-main"
  cp -a "$ROOT/IEEE_TIP_UIU-Net-main/logs" "$EXPORT_DIR/IEEE_TIP_UIU-Net-main/"
  echo "[OK] IEEE_TIP_UIU-Net-main/logs"
fi

# ----- SCTransNet -----
echo "----- SCTransNet -----"
SC_LOG_SRC="$ROOT/SCTransNet-main/log"
SC_LOG_DST="$EXPORT_DIR/SCTransNet-main/log"
mkdir -p "$SC_LOG_DST"

if [[ -d "$SC_LOG_SRC" ]]; then
  # 训练文本日志
  shopt -s nullglob
  txts=("$SC_LOG_SRC"/*.txt)
  if ((${#txts[@]})); then
    cp -a "${txts[@]}" "$SC_LOG_DST/"
    echo "[OK] SCTransNet log/*.txt (${#txts[@]} files)"
  fi
  shopt -u nullglob

  for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
    src="$SC_LOG_SRC/$ds"
    dst="$SC_LOG_DST/$ds"
    [[ -d "$src" ]] || { echo "[SKIP] SCTrans $ds missing"; continue; }
    mkdir -p "$dst"
    if [[ -f "$src/SCTransNet_1000.pth.tar" ]]; then
      cp -a "$src/SCTransNet_1000.pth.tar" "$dst/"
      echo "[OK] SCTrans $ds/SCTransNet_1000.pth.tar"
    else
      echo "[WARN] SCTrans $ds missing SCTransNet_1000.pth.tar"
    fi
    latest_best="$(ls -1t "$src"/SCTransNet_*_best.pth.tar 2>/dev/null | head -1 || true)"
    if [[ -n "${latest_best}" ]]; then
      cp -a "$latest_best" "$dst/"
      echo "[OK] SCTrans $ds/$(basename "$latest_best") (latest best)"
    else
      echo "[WARN] SCTrans $ds no *_best.pth.tar"
    fi
  done
else
  echo "[SKIP] $SC_LOG_SRC missing"
fi

if [[ -d "$ROOT/SCTransNet-main/otherlogs" ]]; then
  mkdir -p "$EXPORT_DIR/SCTransNet-main"
  cp -a "$ROOT/SCTransNet-main/otherlogs" "$EXPORT_DIR/SCTransNet-main/"
  echo "[OK] SCTransNet-main/otherlogs"
fi

# ----- manifest -----
{
  echo "created_at=$(date -Iseconds)"
  echo "root=$ROOT"
  echo "note=slim export: UIU best-only; SCTrans 1000+latest best; MSHNet full weight dirs"
  echo
  find "$EXPORT_DIR" -type f | sed "s|^$EXPORT_DIR/||" | sort
} > "$MANIFEST"

echo "----- sizes -----"
du -sh "$EXPORT_DIR"/* 2>/dev/null || true
du -sh "$EXPORT_DIR"

echo "----- packing gzip -9 (may take a few minutes) -----"
tar -C "$EXPORT_DIR" -cf - . | gzip -9 > "$TAR_GZ"
ls -lh "$TAR_GZ"

# 方便固定文件名下载
LATEST_LINK="${OUT_BASE}/train_artifacts_slim.tar.gz"
ln -sfn "$(basename "$TAR_GZ")" "$LATEST_LINK"
echo "latest symlink: $LATEST_LINK -> $(basename "$TAR_GZ")"

echo "======== export slim done $(date) ========"
echo "下载文件: $TAR_GZ"
echo "或: $LATEST_LINK"
echo "本机解压到仓库根目录示例:"
echo "  tar -xzf train_artifacts_slim.tar.gz -C /path/to/Infrared-detection-model"
echo "勿 git add 权重；仅用本压缩包做本机验收。"
