#!/usr/bin/env python3
"""
用与主表 Pd/Fa 相同口径复测 L2SKNet best_miou.pth.tar。

口径：
  - 概率图 thr=0.5 二值化
  - 连通域质心距离 < 3 → Pd / Fa
  - 附带全局像素 IoU

用法（建议在 AutoDL，有 CUDA）:
  cd ~/autodl-tmp/Infrared-detection-model
  python scripts/eval_l2sk_unified.py \\
    --l2sk-root \"Lightweight model comparison/L2SKNet\" \\
    --data-root RDANet-main/datasets \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure
from tqdm import tqdm

DATASETS = (
    ("NUAA-SIRST", "NUAA-SIRST"),
    ("NUDT-SIRST", "NUDT-SIRST"),
    ("IRSTD-1K", "IRSTD-1k"),  # L2SK 目录名 -> RDANet 目录名
    ("SIRST-V2", "SIRST-V2"),
)


class UnifiedPDFA:
    def __init__(self, match_dist: float = 3.0):
        self.match_dist = float(match_dist)
        self.pd_hits = 0
        self.targets = 0
        self.fa_pixels = 0
        self.all_pixels = 0
        self.inter = 0.0
        self.union = 0.0
        self.sample_ious: List[float] = []

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> None:
        pred = (pred_bin > 0).astype(np.uint8)
        gt = (gt_bin > 0).astype(np.uint8)
        inter = float((pred & gt).sum())
        union = float((pred | gt).sum())
        self.inter += inter
        self.union += union
        self.sample_ious.append(inter / union if union > 0 else 1.0)
        h, w = pred.shape
        self.all_pixels += h * w
        pred_props = measure.regionprops(measure.label(pred, connectivity=2))
        gt_props = measure.regionprops(measure.label(gt, connectivity=2))
        self.targets += len(gt_props)
        remaining = list(pred_props)
        for g in gt_props:
            g_c = np.array(g.centroid)
            hit_i = None
            for i, p in enumerate(remaining):
                if float(np.linalg.norm(np.array(p.centroid) - g_c)) < self.match_dist:
                    hit_i = i
                    self.pd_hits += 1
                    break
            if hit_i is not None:
                remaining.pop(hit_i)
        self.fa_pixels += sum(float(p.area) for p in remaining)

    def summary(self) -> Dict[str, float]:
        return {
            "IoU": float(self.inter / self.union) if self.union > 0 else 0.0,
            "nIoU": float(np.mean(self.sample_ious)) if self.sample_ious else 0.0,
            "Pd": float(self.pd_hits / self.targets) if self.targets > 0 else 0.0,
            "Fa": float(self.fa_pixels / self.all_pixels) if self.all_pixels > 0 else 0.0,
            "Fa_1e6": float(self.fa_pixels / self.all_pixels * 1e6) if self.all_pixels else 0.0,
            "n_images": float(len(self.sample_ious)),
        }


def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0)
    return x, h, w


def list_ids(txt: Path) -> List[str]:
    return [Path(l.strip()).stem for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]


def resolve_img(images: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".bmp", ".tif"):
        p = images / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(images / stem)


def resolve_mask(masks: Path, stem: str) -> Path:
    for name in (f"{stem}.png", f"{stem}_pixels0.png", f"{stem}_mask.png"):
        p = masks / name
        if p.is_file():
            return p
    raise FileNotFoundError(masks / stem)


def load_gray_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, ...]  # 1x1xHxW


def load_mask(path: Path, hw: Tuple[int, int]) -> np.ndarray:
    m = Image.open(path).convert("L")
    if m.size != (hw[1], hw[0]):
        m = m.resize((hw[1], hw[0]), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def load_l2sk(l2sk_root: Path, ckpt: Path, device: str):
    sys.path.insert(0, str(l2sk_root))
    from net import Net

    net = Net(model_name="L2SKNet_UNet")
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def eval_one(net, ds_l2sk: str, ds_rda: str, l2sk_root: Path, data_root: Path, device: str, thr: float):
    # prefer RDANet images; fall back to L2SK data junction
    rda = data_root / ds_rda
    local = l2sk_root / "data" / ds_l2sk
    if (rda / "images").is_dir():
        images, masks = rda / "images", rda / "masks"
        # idx: L2SK img_idx naming
        idx = local / "img_idx" / f"test_{ds_l2sk}.txt"
        if not idx.is_file():
            idx = rda / "test.txt"
    else:
        images, masks = local / "images", local / "masks"
        idx = local / "img_idx" / f"test_{ds_l2sk}.txt"
    ids = list_ids(idx)
    metric = UnifiedPDFA(3.0)
    for stem in tqdm(ids, desc=ds_l2sk, leave=False):
        img_p = resolve_img(images, stem)
        mask_p = resolve_mask(masks, stem)
        x = load_gray_tensor(img_p).to(device)
        _, h0, w0 = x.shape[0], x.shape[2], x.shape[3]
        gt = load_mask(mask_p, (h0, w0))
        with torch.no_grad():
            x_pad, h, w = pad_to_multiple(x, 32)
            prob = net(x_pad)
            if isinstance(prob, (list, tuple)):
                prob = prob[0]
            prob = prob[:, :, :h, :w].squeeze().float().cpu().numpy()
        pred = (prob > thr).astype(np.uint8)
        metric.update(pred, gt)
    return metric.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2sk-root", type=str, default="Lightweight model comparison/L2SKNet")
    ap.add_argument("--data-root", type=str, default="RDANet-main/datasets")
    ap.add_argument("--out-dir", type=str, default="results/ex04_baselines")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument(
        "--datasets",
        type=str,
        default="NUAA-SIRST,NUDT-SIRST,IRSTD-1K,SIRST-V2",
    )
    args = ap.parse_args()

    root = Path(".").resolve()
    l2sk_root = (root / args.l2sk_root).resolve()
    data_root = (root / args.data_root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    want = {x.strip() for x in args.datasets.split(",") if x.strip()}
    rows = []
    for ds_l2sk, ds_rda in DATASETS:
        if ds_l2sk not in want:
            continue
        ckpt = l2sk_root / "log" / ds_l2sk / "L2SKNet_UNet" / "best_miou.pth.tar"
        if not ckpt.is_file():
            print(f"[skip] missing {ckpt}")
            continue
        print(f"\n=== {ds_l2sk} <- {ckpt} ===")
        net = load_l2sk(l2sk_root, ckpt, args.device)
        s = eval_one(net, ds_l2sk, ds_rda, l2sk_root, data_root, args.device, args.thr)
        row = {
            "model": "L2SKNet_UNet",
            "dataset": ds_l2sk,
            "ckpt": str(ckpt),
            "thr": args.thr,
            **s,
        }
        rows.append(row)
        print(
            f"  IoU={100*s['IoU']:.2f}%  nIoU={100*s['nIoU']:.2f}%  "
            f"Pd={100*s['Pd']:.2f}%  Fa={s['Fa_1e6']:.2f}e-6  (N={int(s['n_images'])})"
        )
        del net
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"l2sk_unified_{stamp}.csv"
    latest = out_dir / "l2sk_unified_latest.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        latest.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        (out_dir / f"l2sk_unified_{stamp}.json").write_text(
            json.dumps({"created": stamp, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nsaved {csv_path}")
        print(f"saved {latest}")
    else:
        print("no results")


if __name__ == "__main__":
    main()
