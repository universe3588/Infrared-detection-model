#!/usr/bin/env python3
"""
把 RDANet 四数据集接到 L2SKNet/data 下，供训练使用。

布局（与官方一致）：
  data/<DS>/{images,masks,img_idx}

对 NUDT-SIRST / IRSTD-1K：保留仓库自带 img_idx 划分（与 RDANet 一致），
只把 images/masks 做成目录联接（junction / symlink）。

对 NUAA-SIRST / SIRST-V2：从 RDANet 的 trainval.txt / test.txt 生成 img_idx，
并同样联接 images/masks。

用法（在 L2SKNet 目录或任意位置）：
  python prepare_data.py
  python prepare_data.py --rda-root "D:/Desktop/红外检测模型/RDANet-main/datasets"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
# 本地默认；AutoDL 请显式传 --rda-root /root/autodl-tmp/datasets
_CANDIDATES = [
    HERE.parents[1] / "RDANet-main" / "datasets",
    Path("/root/autodl-tmp/datasets"),
    Path("/root/autodl-tmp/RDANet-main/datasets"),
]
DEFAULT_RDA = next((p for p in _CANDIDATES if p.is_dir()), _CANDIDATES[0])

# L2SK 目录名 -> RDANet 目录名
MAP = {
    "NUDT-SIRST": "NUDT-SIRST",
    "IRSTD-1K": "IRSTD-1k",
    "NUAA-SIRST": "NUAA-SIRST",
    "SIRST-V2": "SIRST-V2",
}

# 官方命名的 idx 文件（已存在则不覆盖）
OFFICIAL_IDX = {
    "NUDT-SIRST": ("train_NUDT-SIRST.txt", "test_NUDT-SIRST.txt"),
    "IRSTD-1K": ("train_IRSTD-1K.txt", "test_IRSTD-1K.txt"),
}


def _is_link_or_junction(p: Path) -> bool:
    try:
        return p.is_symlink() or bool(os.path.islink(p))
    except Exception:
        return False


def link_dir(src: Path, dst: Path) -> None:
    """优先 junction（Windows 无需管理员），失败再试 symlink。"""
    if dst.exists() or dst.is_symlink():
        # 已是指向正确位置则跳过
        try:
            if dst.resolve() == src.resolve():
                print(f"  skip link (exists): {dst}")
                return
        except Exception:
            pass
        if dst.is_dir() and not _is_link_or_junction(dst):
            # 非空真目录：不擅自删除
            n = sum(1 for _ in dst.iterdir())
            if n > 0:
                print(f"  WARN keep existing dir: {dst} ({n} entries)")
                return
            dst.rmdir()
        else:
            if dst.is_symlink() or _is_link_or_junction(dst):
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)

    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        # mklink /J dst src
        cmd = ["cmd", "/c", "mklink", "/J", str(dst), str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise RuntimeError(f"junction failed: {dst} -> {src}")
        print(f"  junction: {dst} -> {src}")
    else:
        os.symlink(src, dst, target_is_directory=True)
        print(f"  symlink: {dst} -> {src}")


def write_idx_from_rda(rda_ds: Path, out_dir: Path, train_name: str, test_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def stems(txt: Path):
        lines = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            lines.append(Path(s).stem)
        return lines

    train_src = rda_ds / "trainval.txt"
    test_src = rda_ds / "test.txt"
    if not train_src.is_file() or not test_src.is_file():
        raise FileNotFoundError(f"missing split under {rda_ds}")

    train_ids = stems(train_src)
    test_ids = stems(test_src)
    (out_dir / train_name).write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out_dir / test_name).write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    print(f"  idx {train_name}: {len(train_ids)}  {test_name}: {len(test_ids)}")


def prepare_one(name: str, rda_root: Path, data_root: Path) -> None:
    rda_name = MAP[name]
    rda_ds = rda_root / rda_name
    if not rda_ds.is_dir():
        raise FileNotFoundError(rda_ds)

    dst = data_root / name
    dst.mkdir(parents=True, exist_ok=True)
    print(f"\n[{name}] <- {rda_ds}")

    link_dir(rda_ds / "images", dst / "images")
    link_dir(rda_ds / "masks", dst / "masks")

    idx_dir = dst / "img_idx"
    idx_dir.mkdir(parents=True, exist_ok=True)

    if name in OFFICIAL_IDX:
        train_n, test_n = OFFICIAL_IDX[name]
        # 官方 idx 已在仓库里；缺则从 RDANet 生成
        if not (idx_dir / train_n).is_file() or not (idx_dir / test_n).is_file():
            write_idx_from_rda(rda_ds, idx_dir, train_n, test_n)
        else:
            nt = sum(1 for x in (idx_dir / train_n).read_text(encoding="utf-8").splitlines() if x.strip())
            ne = sum(1 for x in (idx_dir / test_n).read_text(encoding="utf-8").splitlines() if x.strip())
            print(f"  keep official idx: train={nt} test={ne}")
    else:
        # NUAA / SIRST-V2：统一用 train_*.txt / test_*.txt
        write_idx_from_rda(rda_ds, idx_dir, f"train_{name}.txt", f"test_{name}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rda-root", type=str, default=str(DEFAULT_RDA))
    ap.add_argument("--data-root", type=str, default=str(HERE / "data"))
    ap.add_argument(
        "--datasets",
        type=str,
        default="NUDT-SIRST,IRSTD-1K,NUAA-SIRST,SIRST-V2",
        help="comma list; sirst_aug 不在 RDANet 内，需自行下载",
    )
    args = ap.parse_args()

    rda_root = Path(args.rda_root)
    data_root = Path(args.data_root)
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]

    print(f"rda_root = {rda_root}")
    print(f"data_root = {data_root}")
    for n in names:
        if n not in MAP:
            raise SystemExit(f"unknown dataset {n}; known={list(MAP)}")
        prepare_one(n, rda_root, data_root)

    print("\nDone. Next:")
    print("  python train_ex04.py --model_names L2SKNet_UNet --dataset_names NUAA-SIRST --batchSize 4")


if __name__ == "__main__":
    main()
