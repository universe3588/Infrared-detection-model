#!/usr/bin/env python3
"""
Ex04：三个有公开权重的轻量基线 × RDANet 四数据集 zero-shot 评测。

模型（均在 NUAA-SIRST 上训练，其它集为零样本）：
  - IRPruneDet  : Lightweight model comparison/IRPruneDet/checkpoints/.../irprunedet_model.pkl
  - LW-IRSTNet  : Lightweight model comparison/LW-IRSTNet/LW_IRST.pkl
  - RepISD-Net  : Lightweight model comparison/RepISD-Net/result/NUAA/deploy.pth.tar

数据集（RDANet 布局 images/ + masks/ + test.txt）：
  NUAA-SIRST, NUDT-SIRST, IRSTD-1k, SIRST-V2

指标（与论文统一口径）：
  - 概率图 thr=0.5 二值化（IRPruneDet/LW 对 logits 做 sigmoid；RepISD 取前景通道，头内已 Sigmoid）
  - 像素 IoU / nIoU
  - 目标级 Pd / Fa（连通域质心距离 < 3）

用法：
  python scripts/eval_ex04_baselines.py
  python scripts/eval_ex04_baselines.py --models IRPruneDet,LW --datasets NUAA-SIRST
  python scripts/eval_ex04_baselines.py --limit 8          # 冒烟
  python scripts/eval_ex04_baselines.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LWC = ROOT / "Lightweight model comparison"
DATA_ROOT_DEFAULT = ROOT / "RDANet-main" / "datasets"
OUT_DIR_DEFAULT = ROOT / "results" / "ex04_baselines"

DATASETS_DEFAULT = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "SIRST-V2")
MODELS_DEFAULT = ("IRPruneDet", "LW", "RepISD")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# RepISD 官方 test 归一化（NUAA 统计）
REPISD_MEAN = (0.4287, 0.4290, 0.4314)
REPISD_STD = (0.2277, 0.2276, 0.2292)

_COLLIDE_PREFIXES = (
    "model",
    "utils",
    "dataset",
    "metrics",
    "metric",
    "convert_tool",
    "irprunedet",
)


@dataclass
class RunResult:
    model: str
    dataset: str
    n_images: int
    IoU: float
    nIoU: float
    Pd: float
    Fa: float
    Fa_1e6: float
    seconds: float
    base_size: int
    ckpt: str
    note: str = ""


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self, match_dist: float = 3.0):
        from skimage import measure

        self.measure = measure
        self.match_dist = float(match_dist)
        self.inter = 0.0
        self.union = 0.0
        self.sample_ious: List[float] = []
        self.pd_hits = 0
        self.targets = 0
        self.fa_pixels = 0
        self.all_pixels = 0

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> None:
        pred = (pred_bin > 0).astype(np.uint8)
        gt = (gt_bin > 0).astype(np.uint8)
        inter = float((pred & gt).sum())
        union = float((pred | gt).sum())
        self.inter += inter
        self.union += union
        self.sample_ious.append(inter / union if union > 0 else (1.0 if inter == 0 else 0.0))

        h, w = pred.shape
        self.all_pixels += h * w

        pred_label = self.measure.label(pred, connectivity=2)
        gt_label = self.measure.label(gt, connectivity=2)
        pred_props = self.measure.regionprops(pred_label)
        gt_props = self.measure.regionprops(gt_label)
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
        iou = self.inter / self.union if self.union > 0 else 0.0
        niou = float(np.mean(self.sample_ious)) if self.sample_ious else 0.0
        pd = self.pd_hits / self.targets if self.targets > 0 else 0.0
        fa = self.fa_pixels / self.all_pixels if self.all_pixels > 0 else 0.0
        return {
            "IoU": float(iou),
            "nIoU": float(niou),
            "Pd": float(pd),
            "Fa": float(fa),
            "Fa_1e6": float(fa * 1e6),
            "n_images": float(len(self.sample_ious)),
        }


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _resolve_mask(masks_dir: Path, stem: str) -> Path:
    candidates = [
        masks_dir / f"{stem}.png",
        masks_dir / f"{stem}_pixels0.png",
        masks_dir / f"{stem}.bmp",
        masks_dir / f"{stem}.tif",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"mask not found for {stem} under {masks_dir}")


def _resolve_image(images_dir: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"image not found for {stem} under {images_dir}")


def list_test_ids(dataset_dir: Path) -> List[str]:
    txt = dataset_dir / "test.txt"
    if not txt.is_file():
        raise FileNotFoundError(txt)
    ids = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        # 允许带后缀
        ids.append(Path(s).stem)
    return ids


def load_pair(
    dataset_dir: Path,
    stem: str,
    base_size: int,
    mean: Sequence[float],
    std: Sequence[float],
):
    from PIL import Image
    import torch
    from torchvision import transforms

    img_path = _resolve_image(dataset_dir / "images", stem)
    mask_path = _resolve_mask(dataset_dir / "masks", stem)

    img = Image.open(img_path).convert("RGB").resize((base_size, base_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((base_size, base_size), Image.NEAREST)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(list(mean), list(std)),
        ]
    )
    x = tfm(img).unsqueeze(0)  # 1x3xH×W
    y = (transforms.ToTensor()(mask) > 0.5).float().squeeze(0).numpy().astype(np.uint8)
    return x, y, stem


# ---------------------------------------------------------------------------
# model loaders
# ---------------------------------------------------------------------------
def _purge_collide() -> None:
    doomed = [k for k in list(sys.modules) if any(k == p or k.startswith(p + ".") for p in _COLLIDE_PREFIXES)]
    for k in doomed:
        del sys.modules[k]


def _prefer_repo(repo: Path) -> None:
    """把目标仓库放到 sys.path 最前，并去掉其它基线仓库，避免同名 model/utils 串台。"""
    repo_s = str(repo.resolve())
    drop_roots = {
        str((LWC / "IRPruneDet").resolve()),
        str((LWC / "LW-IRSTNet").resolve()),
        str((LWC / "RepISD-Net").resolve()),
        str((LWC / "LW-IRSTNet" / "models" / "LW_IRST_Net").resolve()),
    }
    kept = []
    for p in sys.path:
        try:
            pr = str(Path(p).resolve())
        except Exception:
            kept.append(p)
            continue
        if pr in drop_roots:
            continue
        kept.append(p)
    sys.path[:] = [repo_s] + kept


def _stub_nni() -> None:
    """IRPruneDet/model/segmentation.py 顶部强依赖 nni，评测加载权重时不需要真实实现。"""
    import types

    if "nni" in sys.modules:
        return

    def _pkg(name: str):
        m = types.ModuleType(name)
        m.__path__ = []  # mark as package
        sys.modules[name] = m
        return m

    nni = _pkg("nni")
    alg = _pkg("nni.algorithms")
    comp = _pkg("nni.algorithms.compression")
    v2 = _pkg("nni.algorithms.compression.v2")
    pt = _pkg("nni.algorithms.compression.v2.pytorch")
    pruning = _pkg("nni.algorithms.compression.v2.pytorch.pruning")
    pruning.L1NormPruner = object
    pruning.SlimPruner = object

    nni_comp = _pkg("nni.compression")
    nni_pt = _pkg("nni.compression.pytorch")
    nni_pt.ModelSpeedup = object
    utils = _pkg("nni.compression.pytorch.utils")
    counter = _pkg("nni.compression.pytorch.utils.counter")
    counter.count_flops_params = lambda *a, **k: (0, 0, None)

    # silence unused
    _ = (nni, alg, comp, v2, pt, nni_comp, nni_pt, utils)


def load_irprunedet(ckpt: Path, device: str):
    import torch

    _purge_collide()
    _prefer_repo(LWC / "IRPruneDet")
    _stub_nni()
    import model.segmentation  # noqa: F401

    try:
        net = torch.load(str(ckpt), map_location=device, weights_only=False)
    except TypeError:
        net = torch.load(str(ckpt), map_location=device)
    net = net.to(device).eval()
    return net, {"kind": "logits", "base_size": 512, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}


def _stub_torchviz() -> None:
    import types

    if "torchviz" in sys.modules:
        return
    m = types.ModuleType("torchviz")
    m.make_dot = lambda *a, **k: None
    m.make_dot_from_trace = lambda *a, **k: None
    sys.modules["torchviz"] = m


def load_lw(ckpt: Path, device: str):
    import torch

    _purge_collide()
    # pkl 内含完整 Module；类定义在 models/LW_IRST_Net
    _prefer_repo(LWC / "LW-IRSTNet" / "models" / "LW_IRST_Net")
    lw_root = str((LWC / "LW-IRSTNet").resolve())
    if lw_root not in sys.path:
        sys.path.insert(1, lw_root)
    _stub_torchviz()
    # 预导入类名，兼容部分 pkl 的全局引用
    try:
        import LW_IRST_ablation  # noqa: F401
    except Exception:
        pass

    try:
        net = torch.load(str(ckpt), map_location=device, weights_only=False)
    except TypeError:
        net = torch.load(str(ckpt), map_location=device)
    net = net.to(device).eval()
    return net, {"kind": "logits", "base_size": 256, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}


def load_repisd(ckpt: Path, device: str):
    import torch

    _purge_collide()
    _prefer_repo(LWC / "RepISD-Net")

    from convert_tool.convert_CDC import config_model
    from model.RepISD import RepISD

    cdc = config_model("cdc")
    net = RepISD(cdc[0], bins=(1, 2, 3, 6), dropout=0.1, classes=2, convert=True, deploy=True)
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()
    return net, {
        "kind": "repisd",
        "base_size": 256,
        "mean": REPISD_MEAN,
        "std": REPISD_STD,
    }


def default_ckpt(name: str) -> Path:
    mapping = {
        "IRPruneDet": LWC / "IRPruneDet" / "checkpoints" / "hard_finetuned" / "irprunedet_model.pkl",
        "LW": LWC / "LW-IRSTNet" / "LW_IRST.pkl",
        "RepISD": LWC / "RepISD-Net" / "result" / "NUAA" / "deploy.pth.tar",
    }
    return mapping[name]


def load_model(name: str, ckpt: Path, device: str):
    loaders = {
        "IRPruneDet": load_irprunedet,
        "LW": load_lw,
        "RepISD": load_repisd,
    }
    if name not in loaders:
        raise KeyError(f"unknown model {name}")
    return loaders[name](ckpt, device)


def predict_prob(net, x, kind: str):
    import torch

    with torch.no_grad():
        out = net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if kind == "logits":
            # IRPruneDet / LW：单通道 logits
            if out.shape[1] > 1:
                out = out[:, 1:2]
            prob = torch.sigmoid(out)
        elif kind == "repisd":
            # 头内已 Sigmoid，取前景通道
            prob = out[:, 1:2]
        else:
            prob = out
        return prob.squeeze().detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# evaluate one model × one dataset
# ---------------------------------------------------------------------------
def eval_one(
    model_name: str,
    net,
    meta: dict,
    dataset: str,
    data_root: Path,
    device: str,
    limit: Optional[int],
    thr: float,
) -> RunResult:
    from tqdm import tqdm
    import torch

    ds_dir = data_root / dataset
    ids = list_test_ids(ds_dir)
    if limit is not None:
        ids = ids[: int(limit)]

    metrics = Metrics(match_dist=3.0)
    t0 = time.time()
    for stem in tqdm(ids, desc=f"{model_name}|{dataset}", leave=False):
        x, gt, _ = load_pair(
            ds_dir,
            stem,
            base_size=int(meta["base_size"]),
            mean=meta["mean"],
            std=meta["std"],
        )
        x = x.to(device)
        prob = predict_prob(net, x, meta["kind"])
        pred = (prob > thr).astype(np.uint8)
        if pred.shape != gt.shape:
            # 兜底：最近邻对齐
            from PIL import Image

            pred = np.array(
                Image.fromarray(pred * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            )
            pred = (pred > 127).astype(np.uint8)
        metrics.update(pred, gt)

    s = metrics.summary()
    return RunResult(
        model=model_name,
        dataset=dataset,
        n_images=int(s["n_images"]),
        IoU=s["IoU"],
        nIoU=s["nIoU"],
        Pd=s["Pd"],
        Fa=s["Fa"],
        Fa_1e6=s["Fa_1e6"],
        seconds=time.time() - t0,
        base_size=int(meta["base_size"]),
        ckpt="",
        note="NUAA-trained; non-NUAA = zero-shot",
    )


def save_results(rows: List[RunResult], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ex04_baselines_{stamp}.csv"
    json_path = out_dir / f"ex04_baselines_{stamp}.json"
    latest_csv = out_dir / "ex04_baselines_latest.csv"
    latest_json = out_dir / "ex04_baselines_latest.json"

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    payload = {
        "created": stamp,
        "rows": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return csv_path, json_path


def print_table(rows: List[RunResult]) -> None:
    if not rows:
        print("(no results)")
        return
    header = f"{'Model':<12} {'Dataset':<12} {'N':>5} {'IoU%':>8} {'nIoU%':>8} {'Pd%':>8} {'Fa(1e-6)':>10} {'sec':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model:<12} {r.dataset:<12} {r.n_images:>5} "
            f"{100 * r.IoU:>8.2f} {100 * r.nIoU:>8.2f} {100 * r.Pd:>8.2f} "
            f"{r.Fa_1e6:>10.2f} {r.seconds:>8.1f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Ex04 baseline eval (3 weighted models × 4 datasets)")
    p.add_argument("--root", type=str, default=str(ROOT), help="project root")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument("--models", type=str, default=",".join(MODELS_DEFAULT))
    p.add_argument("--datasets", type=str, default=",".join(DATASETS_DEFAULT))
    p.add_argument("--device", type=str, default="cuda" if _cuda_ok() else "cpu")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="per-dataset image cap (smoke test)")
    p.add_argument("--irprunedet-ckpt", type=str, default="")
    p.add_argument("--lw-ckpt", type=str, default="")
    p.add_argument("--repisd-ckpt", type=str, default="")
    p.add_argument("--dry-run", action="store_true", help="only check paths")
    return p.parse_args()


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main():
    args = parse_args()
    root = Path(args.root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    ckpt_overrides = {
        "IRPruneDet": Path(args.irprunedet_ckpt) if args.irprunedet_ckpt else default_ckpt("IRPruneDet"),
        "LW": Path(args.lw_ckpt) if args.lw_ckpt else default_ckpt("LW"),
        "RepISD": Path(args.repisd_ckpt) if args.repisd_ckpt else default_ckpt("RepISD"),
    }

    print(f"device={args.device}  thr={args.thr}  limit={args.limit}")
    print(f"data_root={data_root}")
    for m in models:
        ck = ckpt_overrides[m]
        ok = ck.is_file()
        print(f"  ckpt[{m}] {'OK' if ok else 'MISSING'}  {ck}")
        if not ok and not args.dry_run:
            raise FileNotFoundError(ck)
    for d in datasets:
        ds = data_root / d
        n = len(list_test_ids(ds)) if ds.is_dir() else -1
        print(f"  dataset[{d}] n_test={n}  {ds}")
        if n < 0 and not args.dry_run:
            raise FileNotFoundError(ds)

    if args.dry_run:
        print("dry-run done.")
        return

    rows: List[RunResult] = []
    for m in models:
        ckpt = ckpt_overrides[m]
        print(f"\n=== load {m} ===")
        net, meta = load_model(m, ckpt, args.device)
        for d in datasets:
            r = eval_one(
                model_name=m,
                net=net,
                meta=meta,
                dataset=d,
                data_root=data_root,
                device=args.device,
                limit=args.limit,
                thr=args.thr,
            )
            r.ckpt = str(ckpt)
            rows.append(r)
            print(
                f"  {m} @ {d}: IoU={100*r.IoU:.2f}%  nIoU={100*r.nIoU:.2f}%  "
                f"Pd={100*r.Pd:.2f}%  Fa={r.Fa_1e6:.2f}e-6  ({r.n_images} img, {r.seconds:.1f}s)"
            )
        # 释放后再载下一个，避免路径串台
        del net
        _purge_collide()

    print("\n==== Summary ====")
    print_table(rows)
    csv_path, json_path = save_results(rows, out_dir)
    print(f"\nsaved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"latest: {out_dir / 'ex04_baselines_latest.csv'}")


if __name__ == "__main__":
    main()
