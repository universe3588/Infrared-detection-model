#!/usr/bin/env python3
"""
统一补齐教师/学生目标级 P_d / F_a（Ex05b / 附录 P0）。

同一套口径（与 SCTrans 官方评测一致）：
  1) 得到概率图（DNA/MSH: sigmoid(logits)；UIU/SCTrans: 网络已输出概率）
  2) 固定阈值 threshold=0.5 二值化
  3) 连通域质心距离 < 3 判为命中
  4) P_d = 命中目标数 / GT 目标数
     F_a = 未匹配预测像素数 / 全图像素数

同时附带像素 IoU，便于与主表 IoU 交叉核对。

用法（在项目根或任意位置）：
  # 自动发现四骨干 × 四数据集 × 师生，一次跑完
  python scripts/eval_pd_fa_unified.py --root /path/to/红外检测模型

  # 只跑某骨干 / 某数据集
  python scripts/eval_pd_fa_unified.py --backbone dna,sctrans --datasets NUAA-SIRST,NUDT-SIRST

  # 手工指定 manifest（覆盖自动发现）
  python scripts/eval_pd_fa_unified.py --manifest scripts/pd_fa_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATASETS_DEFAULT = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "SIRST-V2")
BACKBONES_DEFAULT = ("dna", "uiu", "msh", "sctrans")


def _require_runtime():
    """延迟导入重依赖，便于 --dry_run 在无 GPU 环境核对路径。"""
    global np, torch, measure, tqdm
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from skimage import measure  # noqa: F401
    from tqdm import tqdm  # noqa: F401
    return np, torch, measure, tqdm

REPO_REL = {
    "dna": "Infrared-Small-Target-Detection-master",
    "uiu": "IEEE_TIP_UIU-Net-main",
    "msh": "MSHNet-main",
    "sctrans": "SCTransNet-main",
}

# 切换骨干时需要从 sys.modules 清掉的前缀，避免同名包串台
_COLLIDE_PREFIXES = (
    "model",
    "utils",
    "metrics",
    "metric",
    "compress",
    "dataset",
    "data_loader",
    "train_irstd",
    "warmup_scheduler",
)


# ---------------------------------------------------------------------------
# 统一 P_d / F_a
# ---------------------------------------------------------------------------
class UnifiedPDFA:
    """Centroid-match P_d / F_a at a fixed binary mask (threshold already applied)."""

    def __init__(self, match_dist: float = 3.0):
        self.match_dist = float(match_dist)
        self.pd_hits = 0
        self.targets = 0
        self.fa_pixels = 0
        self.all_pixels = 0
        self.inter = 0.0
        self.union = 0.0

    def update(self, pred_bin, gt_bin):
        """pred_bin / gt_bin: 2D {0,1} arrays, same H×W."""
        np, _, measure, _ = _require_runtime()
        pred_bin = (pred_bin > 0).astype(np.uint8)
        gt_bin = (gt_bin > 0).astype(np.uint8)
        assert pred_bin.ndim == 2 and gt_bin.ndim == 2
        assert pred_bin.shape == gt_bin.shape

        h, w = pred_bin.shape
        self.all_pixels += h * w
        inter = float((pred_bin & gt_bin).sum())
        union = float((pred_bin | gt_bin).sum())
        self.inter += inter
        self.union += union

        pred_label = measure.label(pred_bin, connectivity=2)
        gt_label = measure.label(gt_bin, connectivity=2)
        pred_props = measure.regionprops(pred_label)
        gt_props = measure.regionprops(gt_label)

        self.targets += len(gt_props)
        remaining = list(pred_props)

        for g in gt_props:
            g_c = np.array(g.centroid)
            hit_i = None
            for i, p in enumerate(remaining):
                dist = float(np.linalg.norm(np.array(p.centroid) - g_c))
                if dist < self.match_dist:
                    hit_i = i
                    self.pd_hits += 1
                    break
            if hit_i is not None:
                remaining.pop(hit_i)

        # 未匹配预测连通域面积之和 = 虚警像素（与 DNA/SCTrans 惯例一致）
        self.fa_pixels += sum(float(p.area) for p in remaining)

    def get(self) -> Dict[str, float]:
        pd = float(self.pd_hits / self.targets) if self.targets > 0 else 0.0
        fa = float(self.fa_pixels / self.all_pixels) if self.all_pixels > 0 else 0.0
        iou = float(self.inter / self.union) if self.union > 0 else 0.0
        return {
            "Pd": pd,
            "Fa": fa,
            "IoU": iou,
            "targets": float(self.targets),
            "pd_hits": float(self.pd_hits),
            "fa_pixels": float(self.fa_pixels),
            "pixels": float(self.all_pixels),
        }


def to_prob(pred, kind: str):
    """kind: 'logits' | 'prob'."""
    if kind == "logits":
        import torch
        return torch.sigmoid(pred)
    return pred


def _as_numpy_2d(t):
    import numpy as np
    x = t.detach().float().cpu().numpy()
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"expect HxW after squeeze, got {x.shape}")
    return x


# ---------------------------------------------------------------------------
# 路径发现
# ---------------------------------------------------------------------------
def _newest(paths: Sequence[str]) -> Optional[str]:
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _glob(root: str, pattern: str) -> List[str]:
    return glob.glob(os.path.join(root, pattern), recursive=True)


def pick_student(repo: str, dataset: str) -> Optional[str]:
    """优先正式 {DS}_pipeline_*/student_best.pth.tar（跳过 ablation/repair）。"""
    cands = _glob(repo, os.path.join("**", f"{dataset}_pipeline_*", "student_best.pth.tar"))
    if not cands:
        return None
    scored: List[Tuple[int, float, str]] = []
    for p in cands:
        norm = p.replace("\\", "/").lower()
        score = 0
        # 排除消融 / 临时修复目录
        if any(tok in norm for tok in ("ablation", "repair", "fair_reeval", "/tmp/", "debug")):
            score -= 100
        # 更靠近 compress_runs/<DS>_pipeline_* 的路径加分（避免嵌套副本抢最新）
        parts = norm.split("/")
        try:
            i = parts.index("compress_runs")
            # compress_runs / <ds>_pipeline_xxx / student_best  => 理想深度
            if i + 2 < len(parts) and parts[i + 1].startswith(dataset.lower()):
                score += 20
            else:
                score += 5
        except ValueError:
            pass
        args_json = os.path.join(os.path.dirname(p), "args.json")
        if os.path.isfile(args_json):
            try:
                with open(args_json, "r", encoding="utf-8") as f:
                    aj = json.load(f)
                rates = str(aj.get("prune_rates", ""))
                # SCTrans v3
                if "0.05" in rates and "0.15" in rates and "0.25" in rates:
                    score += 15
                # 正式主实验常见配额
                if aj.get("mode") == "pipeline":
                    score += 3
                # 有记录 best_iou 的完整蒸馏产物更可信（文件本身也会有）
            except Exception:
                pass
        scored.append((score, os.path.getmtime(p), p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0]
    if best[0] < 0:
        return None
    return best[2]


def pick_teacher(backbone: str, repo: str, dataset: str) -> Optional[str]:
    if backbone == "dna":
        return _newest(_glob(repo, os.path.join("result", f"{dataset}_DNANet_*_wDS", "mIoU__DNANet_*_epoch.pth.tar")))
    if backbone == "uiu":
        return _newest(_glob(repo, os.path.join("saved_models", "uiunet_irstd", f"{dataset}_*", "uiunet_best.pth.tar")))
    if backbone == "msh":
        return _newest(_glob(repo, os.path.join("weight", f"MSHNet-{dataset}-*", "weight.pkl")))
    if backbone == "sctrans":
        return _newest(_glob(repo, os.path.join("log", dataset, "SCTransNet_*_best.pth.tar")))
    return None


@dataclass
class Job:
    backbone: str
    dataset: str
    role: str  # teacher | student
    ckpt: str
    repo: str


def discover_jobs(
    root: str,
    backbones: Sequence[str],
    datasets: Sequence[str],
    roles: Sequence[str],
    manifest: Optional[Dict[str, Any]] = None,
) -> List[Job]:
    jobs: List[Job] = []
    man_jobs = []
    if manifest and "jobs" in manifest:
        man_jobs = manifest["jobs"]

    def man_override(bb: str, ds: str, role: str) -> Optional[str]:
        for j in man_jobs:
            if j.get("backbone") == bb and j.get("dataset") == ds and j.get("role") == role:
                return j.get("ckpt")
        # also allow nested dict: manifest[bb][ds][role]
        block = (manifest or {}).get(bb, {})
        if isinstance(block, dict):
            ds_block = block.get(ds, {})
            if isinstance(ds_block, dict) and role in ds_block:
                return ds_block[role]
        return None

    for bb in backbones:
        repo = os.path.join(root, REPO_REL[bb])
        if not os.path.isdir(repo):
            print(f"[warn] missing repo: {repo}")
            continue
        for ds in datasets:
            for role in roles:
                ckpt = man_override(bb, ds, role)
                if not ckpt:
                    if role == "teacher":
                        ckpt = pick_teacher(bb, repo, ds)
                    else:
                        ckpt = pick_student(repo, ds)
                if not ckpt:
                    print(f"[skip] no ckpt for {bb}/{ds}/{role}")
                    continue
                if not os.path.isabs(ckpt):
                    # relative to repo first, then root
                    cand = os.path.join(repo, ckpt)
                    if not os.path.isfile(cand):
                        cand = os.path.join(root, ckpt)
                    ckpt = cand
                if not os.path.isfile(ckpt):
                    print(f"[skip] missing file {bb}/{ds}/{role}: {ckpt}")
                    continue
                jobs.append(Job(backbone=bb, dataset=ds, role=role, ckpt=ckpt, repo=repo))
    return jobs


# ---------------------------------------------------------------------------
# 各骨干：加载 / 前向 / loader
# ---------------------------------------------------------------------------
@contextmanager
def enter_repo(repo: str):
    repo = os.path.abspath(repo)
    # 清冲突模块
    doomed = [k for k in list(sys.modules) if k == "model" or k.startswith(_COLLIDE_PREFIXES)]
    for k in doomed:
        sys.modules.pop(k, None)
    old_path = list(sys.path)
    old_cwd = os.getcwd()
    sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        doomed = [k for k in list(sys.modules) if k == "model" or k.startswith(_COLLIDE_PREFIXES)]
        for k in doomed:
            sys.modules.pop(k, None)


def _load_state(ckpt_path: str):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                state = {k.replace("module.", ""): v for k, v in state.items()}
                return state, ckpt
        sample = next(iter(ckpt.values()))
        if torch.is_tensor(sample):
            state = {k.replace("module.", ""): v for k, v in ckpt.items()}
            return state, {}
    raise ValueError(f"cannot parse state_dict: {ckpt_path}")


def _strip_sctrans(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def build_loader_dna(dataset: str, dataset_dir: str, base_size: int = 256):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from model.load_param_data import load_dataset
    from model.utils import TestSetLoader

    root = dataset_dir
    dataset_path = os.path.join(root, dataset)
    _, val_ids, _ = load_dataset(root, dataset, "50_50", subset_size=0, subset_seed=42)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    testset = TestSetLoader(
        dataset_path, img_id=val_ids, base_size=base_size, crop_size=base_size,
        transform=tf, suffix=".png",
    )
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return loader, "logits"


def build_model_dna(ckpt: str, role: str, device):
    from model.load_param_data import load_param
    from model.model_DNANet import DNANet, Res_CBAM_block

    nb_filter, num_blocks = load_param("three", "resnet_18")
    state, meta = _load_state(ckpt)
    if role == "student":
        nb = meta.get("nb_filter")
        if not nb:
            raise ValueError(f"DNA student ckpt missing nb_filter: {ckpt}")
        nb_filter = list(nb)
    model = DNANet(
        num_classes=1, input_channels=3, block=Res_CBAM_block,
        num_blocks=num_blocks, nb_filter=list(nb_filter), deep_supervision=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] dna missing {len(missing)} keys")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def iter_dna(model, loader, device, threshold: float):
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        preds = model(data)
        pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
        prob = to_prob(pred, "logits")
        pb = (prob[0, 0] > threshold)
        gb = (labels[0, 0] > 0.5)
        yield _as_numpy_2d(pb), _as_numpy_2d(gb)


def build_loader_uiu(dataset: str, dataset_dir: str, rescale: int = 320):
    from types import SimpleNamespace
    from train_irstd import build_loaders

    root = os.path.join(dataset_dir, dataset)
    args = SimpleNamespace(rescale=rescale, crop=288, batch_size=1, workers=0)
    _, test_loader = build_loaders(args, root)
    return test_loader, "prob"


def build_model_uiu(ckpt: str, role: str, device):
    from model import UIUNET

    state, meta = _load_state(ckpt)
    if role == "student":
        enc = meta.get("enc_outs")
        dec = meta.get("dec_outs")
        me = meta.get("mids_enc")
        md = meta.get("mids_dec")
        if not enc or not dec:
            raise ValueError(f"UIU student ckpt missing enc/dec outs: {ckpt}")
        model = UIUNET(3, 1, enc_outs=list(enc), dec_outs=list(dec),
                       mids_enc=list(me), mids_dec=list(md))
    else:
        model = UIUNET(3, 1)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] uiu missing {len(missing)} keys")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def iter_uiu(model, loader, device, threshold: float):
    for data in loader:
        inputs = data["image"].float().to(device)
        labels = data["label"].float().to(device)
        d0, *_ = model(inputs)
        prob = to_prob(d0, "prob")
        pb = (prob[0, 0] > threshold)
        gb = (labels[0, 0] > 0.5)
        yield _as_numpy_2d(pb), _as_numpy_2d(gb)


def build_loader_msh(dataset: str, dataset_dir: str, base_size: int = 256):
    from types import SimpleNamespace
    import torch.utils.data as Data
    from utils.data import IRSTD_Dataset

    data_args = SimpleNamespace(
        dataset_dir=os.path.join(dataset_dir, dataset),
        crop_size=base_size,
        base_size=base_size,
    )
    valset = IRSTD_Dataset(data_args, mode="val")
    loader = Data.DataLoader(valset, 1, shuffle=False, drop_last=False, num_workers=0)
    return loader, "logits"


def build_model_msh(ckpt: str, role: str, device):
    from model.MSHNet import MSHNet
    from compress.adapters.mshnet_adapter import TEACHER_CHANNELS

    state, meta = _load_state(ckpt)
    if role == "student":
        ch = meta.get("channels") or meta.get("param_channels") or list(TEACHER_CHANNELS)
        model = MSHNet(3, channels=list(ch))
    else:
        model = MSHNet(3)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] msh missing {len(missing)} keys")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def iter_msh(model, loader, device, threshold: float):
    for data, mask in loader:
        data = data.to(device)
        mask = mask.to(device)
        _, pred = model(data, True)
        prob = to_prob(pred, "logits")
        pb = (prob[0, 0] > threshold)
        gb = (mask[0, 0] > 0.5)
        yield _as_numpy_2d(pb), _as_numpy_2d(gb)


def build_loader_sctrans(dataset: str, dataset_dir: str, patch_size: int = 256):
    from torch.utils.data import DataLoader
    from dataset import TestSetLoader

    test_set = TestSetLoader(dataset_dir, dataset, dataset, img_norm_cfg=None)
    loader = DataLoader(test_set, num_workers=0, batch_size=1, shuffle=False)
    return loader, "prob"


def build_model_sctrans(ckpt: str, role: str, device, img_size: int = 256):
    from compress.adapters.sctrans_adapter import TEACHER_CHANNELS
    from model.Config import get_SCTrans_config
    from model.SCTransNet import SCTransNet

    state, meta = _load_state(ckpt)
    state = _strip_sctrans(state)
    if role == "student":
        ch = meta.get("channel_num") or meta.get("channels") or list(TEACHER_CHANNELS)
    else:
        ch = list(TEACHER_CHANNELS)
    cfg = get_SCTrans_config()
    model = SCTransNet(
        cfg, mode="test", deepsuper=True, img_size=img_size, channel_num=list(ch),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] sctrans missing {len(missing)} keys (ok if deepsuper heads differ)")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def iter_sctrans(model, loader, device, threshold: float):
    import torch
    old_mode = getattr(model, "mode", "test")
    model.mode = "test"
    try:
        for batch in loader:
            if len(batch) >= 3:
                img, gt_mask, size = batch[0], batch[1], batch[2]
            else:
                img, gt_mask = batch[0], batch[1]
                size = None
            img = img.to(device)
            pred = model(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            if size is not None:
                # size 可能是 tensor([H,W]) 或 list
                if torch.is_tensor(size):
                    h, w = int(size[0]), int(size[1])
                else:
                    h, w = int(size[0]), int(size[1])
                pred = pred[:, :, :h, :w]
                gt_mask = gt_mask[:, :, :h, :w]
            prob = to_prob(pred, "prob")
            pb = (prob[0, 0] > threshold)
            gb = (gt_mask[0, 0] > 0.5)
            yield _as_numpy_2d(pb), _as_numpy_2d(gb)
    finally:
        model.mode = old_mode


BACKENDS = {
    "dna": (build_loader_dna, build_model_dna, iter_dna),
    "uiu": (build_loader_uiu, build_model_uiu, iter_uiu),
    "msh": (build_loader_msh, build_model_msh, iter_msh),
    "sctrans": (build_loader_sctrans, build_model_sctrans, iter_sctrans),
}


# ---------------------------------------------------------------------------
# 评测一个 job
# ---------------------------------------------------------------------------
@dataclass
class Result:
    backbone: str
    dataset: str
    role: str
    ckpt: str
    Pd: float
    Fa: float
    IoU: float
    n_images: int
    seconds: float
    threshold: float
    match_dist: float
    note: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


def eval_one(
    job: Job,
    dataset_dir: str,
    device,
    threshold: float,
    match_dist: float,
    job_pbar=None,
) -> Result:
    _, torch, _, tqdm = _require_runtime()
    build_loader, build_model, iter_fn = BACKENDS[job.backbone]
    t0 = time.time()
    with enter_repo(job.repo):
        loader, _kind = build_loader(job.dataset, dataset_dir)
        model = build_model(job.ckpt, job.role, device)
        metric = UnifiedPDFA(match_dist=match_dist)
        n = len(loader) if hasattr(loader, "__len__") else None
        desc = f"{job.backbone}/{job.dataset}/{job.role}"
        img_bar = tqdm(
            total=n,
            desc=f"  img {desc}",
            unit="img",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        n_images = 0
        for pred_bin, gt_bin in iter_fn(model, loader, device, threshold):
            metric.update(pred_bin, gt_bin)
            n_images += 1
            stats = metric.get()
            img_bar.set_postfix(
                Pd=f"{stats['Pd']:.4f}",
                Fa=f"{stats['Fa']:.2e}",
                IoU=f"{stats['IoU']:.4f}",
            )
            img_bar.update(1)
            if job_pbar is not None:
                job_pbar.set_postfix_str(
                    f"{desc} Pd={stats['Pd']:.3f} Fa={stats['Fa']:.2e}"
                )
        img_bar.close()
        stats = metric.get()
        # 释放显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dt = time.time() - t0
    return Result(
        backbone=job.backbone,
        dataset=job.dataset,
        role=job.role,
        ckpt=job.ckpt,
        Pd=stats["Pd"],
        Fa=stats["Fa"],
        IoU=stats["IoU"],
        n_images=n_images,
        seconds=dt,
        threshold=threshold,
        match_dist=match_dist,
        note="prob@thr + centroid<match_dist; Fa=unmatched_pred_pixels/all_pixels",
        extra={
            "targets": stats["targets"],
            "pd_hits": stats["pd_hits"],
            "fa_pixels": stats["fa_pixels"],
        },
    )


def fmt_hms(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_outputs(results: List[Result], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"pd_fa_results_{stamp}.json")
    csv_path = os.path.join(out_dir, f"pd_fa_results_{stamp}.csv")
    latest_json = os.path.join(out_dir, "pd_fa_results_latest.json")
    latest_csv = os.path.join(out_dir, "pd_fa_results_latest.csv")
    summary_md = os.path.join(out_dir, "pd_fa_summary_latest.md")

    payload = [asdict(r) for r in results]
    for path in (json_path, latest_json):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    fields = [
        "backbone", "dataset", "role", "Pd", "Fa", "IoU",
        "n_images", "seconds", "threshold", "match_dist", "ckpt",
    ]
    for path in (csv_path, latest_csv):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = asdict(r)
                row["Pd"] = f"{r.Pd:.6f}"
                row["Fa"] = f"{r.Fa:.6e}"
                row["IoU"] = f"{r.IoU:.6f}"
                w.writerow(row)

    # 宽表：师生并排，方便直接贴论文
    keys = sorted({(r.backbone, r.dataset) for r in results})
    lines = [
        "# 统一 P_d / F_a 结果（threshold=0.5, centroid<3）",
        "",
        "| 骨干 | 数据集 | 教师 P_d | 教师 F_a | 教师 IoU | 学生 P_d | 学生 F_a | 学生 IoU |",
        "|------|--------|----------|----------|----------|----------|----------|----------|",
    ]
    by = {(r.backbone, r.dataset, r.role): r for r in results}
    for bb, ds in keys:
        t = by.get((bb, ds, "teacher"))
        s = by.get((bb, ds, "student"))
        def cell(r: Optional[Result], attr: str, fmt: str) -> str:
            if r is None:
                return "--"
            return format(getattr(r, attr), fmt)
        lines.append(
            f"| {bb} | {ds} | "
            f"{cell(t, 'Pd', '.4f')} | {cell(t, 'Fa', '.2e')} | {cell(t, 'IoU', '.4f')} | "
            f"{cell(s, 'Pd', '.4f')} | {cell(s, 'Fa', '.2e')} | {cell(s, 'IoU', '.4f')} |"
        )
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[save] {csv_path}")
    print(f"[save] {json_path}")
    print(f"[save] {summary_md}")
    return latest_csv, summary_md


def parse_args():
    p = argparse.ArgumentParser(description="Unified teacher/student P_d F_a evaluation")
    p.add_argument("--root", type=str, default="",
                   help="项目根（含四模型目录与 RDANet-main/datasets）。默认=本脚本上级")
    p.add_argument("--dataset_dir", type=str, default="",
                   help="数据集根，默认 <root>/RDANet-main/datasets")
    p.add_argument("--backbone", type=str, default="dna,uiu,msh,sctrans",
                   help="逗号分隔: dna,uiu,msh,sctrans")
    p.add_argument("--datasets", type=str, default="NUAA-SIRST,NUDT-SIRST,IRSTD-1k,SIRST-V2")
    p.add_argument("--roles", type=str, default="teacher,student")
    p.add_argument("--manifest", type=str, default="",
                   help="可选 JSON，覆盖自动发现的 ckpt 路径")
    p.add_argument("--out_dir", type=str, default="",
                   help="输出目录，默认 <root>/pd_fa_eval")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--match_dist", type=float, default=3.0)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--dry_run", action="store_true", help="只打印将要评测的 job，不跑推理")
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(args.root or os.path.join(script_dir, ".."))
    dataset_dir = os.path.abspath(args.dataset_dir or os.path.join(root, "RDANet-main", "datasets"))
    out_dir = os.path.abspath(args.out_dir or os.path.join(root, "pd_fa_eval"))

    backbones = [x.strip().lower() for x in args.backbone.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    roles = [x.strip() for x in args.roles.split(",") if x.strip()]
    for bb in backbones:
        if bb not in BACKENDS:
            raise SystemExit(f"unknown backbone: {bb}")

    manifest = None
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    jobs = discover_jobs(root, backbones, datasets, roles, manifest)
    print("======== unified P_d/F_a ========")
    print(f"root={root}")
    print(f"dataset_dir={dataset_dir}")
    print(f"threshold={args.threshold}  match_dist={args.match_dist}")
    print(f"jobs={len(jobs)}  out_dir={out_dir}")
    for j in jobs:
        print(f"  - {j.backbone:8s} {j.dataset:12s} {j.role:7s} <- {j.ckpt}")

    if args.dry_run:
        print("[dry_run] exit without eval")
        return

    if not jobs:
        raise SystemExit("no jobs discovered; check ckpt paths or --manifest")

    _, torch, _, tqdm = _require_runtime()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    results: List[Result] = []
    t_all = time.time()
    job_bar = tqdm(
        total=len(jobs),
        desc="jobs",
        unit="job",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    for idx, job in enumerate(jobs):
        print(f"\n-------- [{idx+1}/{len(jobs)}] {job.backbone}/{job.dataset}/{job.role} --------")
        try:
            res = eval_one(
                job, dataset_dir, device,
                threshold=args.threshold, match_dist=args.match_dist, job_pbar=job_bar,
            )
            results.append(res)
            print(
                f"[ok] Pd={res.Pd:.4f}  Fa={res.Fa:.4e}  IoU={res.IoU:.4f}  "
                f"imgs={res.n_images}  time={fmt_hms(res.seconds)}"
            )
        except Exception as e:
            print(f"[fail] {job.backbone}/{job.dataset}/{job.role}: {e}")
            results.append(Result(
                backbone=job.backbone, dataset=job.dataset, role=job.role, ckpt=job.ckpt,
                Pd=float("nan"), Fa=float("nan"), IoU=float("nan"),
                n_images=0, seconds=0.0, threshold=args.threshold, match_dist=args.match_dist,
                note=f"FAILED: {e}",
            ))
        # 粗 ETA：按已完成 job 平均耗时
        done = idx + 1
        elapsed = time.time() - t_all
        avg = elapsed / done
        remain = avg * (len(jobs) - done)
        job_bar.set_postfix_str(f"ETA_rest≈{fmt_hms(remain)}")
        job_bar.update(1)
        # 增量落盘，中断也不丢
        write_outputs(results, out_dir)

    job_bar.close()
    print(f"\n======== ALL done in {fmt_hms(time.time() - t_all)} ========")
    _, md = write_outputs(results, out_dir)
    print("宽表预览：")
    with open(md, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
