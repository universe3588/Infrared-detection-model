#!/usr/bin/env python3
"""
DNA-Net 通道敲除校准（Ex08 实测）：在教师上按比例逐步屏蔽通道（不重训），
测 IoU，写出 MATLAB `plot_knockout_curves` 可用的 CSV。

曲线含义
  topk_I    : 先去掉 I 最高（重要）的通道 → 预期掉点最快
  bottomk_I : 先去掉 I 最低（冗余）的通道 → 预期掉点最慢
  random    : 随机去掉
  l1        : 按 conv2 权重 L1 范数，先去掉范数最大的通道

示例（在仓库根或任意目录）：
  python scripts/eval_channel_knockout.py \\
    --dnanet-root DNANet \\
    --teacher-ckpt DNANet/result/NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar \\
    --scores-path DNANet/compress_runs/NUAA-SIRST_pipeline_20261029_181406/scores.pt \\
    --dataset NUAA-SIRST \\
    --data-root RDANet-main/datasets \\
    --device cuda \\
    --out "面向边缘部署的红外弱小目标检测可迁移压缩框架/images/Matlab/敲除校准曲线/data/knockout_iou_nuaa.csv"

CPU 冒烟（少比例 + 限图）：
  python scripts/eval_channel_knockout.py ... --device cpu --ratios 0,0.1,0.2,0.3 --limit 16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="DNA-Net progressive channel knockout (Ex08)")
    p.add_argument("--dnanet-root", type=str, default="DNANet")
    p.add_argument("--teacher-ckpt", type=str, required=True)
    p.add_argument("--scores-path", type=str, default="",
                   help="compress 产出的 scores.pt；若空则现场重算 full I（需 GPU 更合适）")
    p.add_argument("--dataset", type=str, default="NUAA-SIRST")
    p.add_argument("--data-root", type=str, default="RDANet-main/datasets")
    p.add_argument("--split-method", type=str, default="50_50")
    p.add_argument("--base-size", type=int, default=256)
    p.add_argument("--ratios", type=str,
                   default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--methods", type=str, default="topk_I,bottomk_I,random,l1",
                   help="逗号分隔：topk_I,bottomk_I,random,l1")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help=">0 时只测前 N 张（冒烟）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.7, help="仅 --scores-path 为空重算时用")
    p.add_argument("--score-batches", type=int, default=50)
    p.add_argument("--channel-size", type=str, default="three")
    p.add_argument("--backbone", type=str, default="resnet_18")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(name)


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state_dict_flexible(ckpt_path: str) -> dict:
    ckpt = torch_load(ckpt_path)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        sample = next(iter(ckpt.values()))
        if torch.is_tensor(sample):
            return ckpt
    raise ValueError(f"Cannot parse state_dict from {ckpt_path}")


def load_scores(path: str) -> Dict[str, torch.Tensor]:
    payload = torch_load(path)
    if isinstance(payload, dict) and "scores" in payload:
        scores = payload["scores"]
    else:
        scores = payload
    if not isinstance(scores, dict):
        raise ValueError("scores.pt must be dict with stage0..stage4")
    out = {}
    for k, v in scores.items():
        key = k if str(k).startswith("stage") else f"stage{k}"
        out[key] = v.detach().float().cpu()
    return out


def choose_drop_indices(
    scores: Dict[str, torch.Tensor],
    remove_ratio: float,
    mode: str,
    seed: int,
) -> Dict[int, torch.Tensor]:
    """Return per-stage channel indices to ZERO (remove)."""
    drop: Dict[int, torch.Tensor] = {}
    for s in range(5):
        key = f"stage{s}"
        I = scores[key].float().cpu()
        c = int(I.numel())
        k = int(round(c * float(remove_ratio)))
        k = max(0, min(k, c))
        if k == 0:
            drop[s] = torch.zeros(0, dtype=torch.long)
            continue
        if mode == "topk_I":
            drop[s] = torch.topk(I, k, largest=True).indices
        elif mode == "bottomk_I":
            drop[s] = torch.topk(I, k, largest=False).indices
        elif mode == "random":
            g = torch.Generator().manual_seed(int(seed) + 1000 * s + int(remove_ratio * 1000))
            drop[s] = torch.randperm(c, generator=g)[:k]
        elif mode == "l1":
            # l1 scores are passed in the same dict under stage*
            drop[s] = torch.topk(I, k, largest=True).indices
        else:
            raise ValueError(f"unknown mode: {mode}")
    return drop


def attach_zero_hooks(model: nn.Module, adapter, drop_idx: Dict[int, torch.Tensor]):
    """Zero selected output channels on every Res_CBAM block of each stage."""
    handles = []
    hook_modules = adapter.get_hook_modules(model)

    for stage, modules in hook_modules.items():
        removed = drop_idx.get(stage, torch.zeros(0, dtype=torch.long))
        if removed.numel() == 0:
            continue
        idx_cpu = removed.long().cpu()

        def make_hook(index_tensor: torch.Tensor):
            def _hook(_module, _inp, out):
                # out: [B, C, H, W]
                idx = index_tensor.to(device=out.device, non_blocking=True)
                out = out.clone()
                out[:, idx] = 0
                return out

            return _hook

        hfn = make_hook(idx_cpu)
        for m in modules:
            handles.append(m.register_forward_hook(hfn))
    return handles


def clear_hooks(handles: List):
    for h in handles:
        h.remove()
    handles.clear()


@torch.no_grad()
def eval_miou(model, loader, device, limit: int = 0) -> float:
    from model.metric import mIoU
    from compress.losses import take_final

    model.eval()
    metric = mIoU(1)
    n = 0
    for data, labels in loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(data)
        pred = take_final(preds)
        metric.update(pred, labels)
        n += data.size(0)
        if limit > 0 and n >= limit:
            break
    _, miou = metric.get()
    return float(miou)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    ratios = [float(x) for x in args.ratios.split(",") if x.strip() != ""]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    launch_cwd = os.getcwd()
    dnanet_root = args.dnanet_root
    if not os.path.isabs(dnanet_root):
        dnanet_root = os.path.abspath(os.path.join(launch_cwd, dnanet_root))
    else:
        dnanet_root = os.path.abspath(dnanet_root)
    if not os.path.isdir(dnanet_root):
        raise SystemExit(f"DNANet root not found: {dnanet_root}")
    workspace = os.path.dirname(dnanet_root)

    def resolve_path(p: str) -> str:
        """Resolve paths before/after chdir; accept DNANet/... or result/... forms."""
        if not p:
            return p
        p_norm = p.replace("\\", "/")
        strip_names = []
        if p_norm.startswith("DNANet/") or p_norm.startswith("DNANet\\"):
            strip_names.append(p_norm.split("/", 1)[-1] if "/" in p_norm else p_norm.split("\\", 1)[-1])
        cands = [
            p,
            os.path.join(launch_cwd, p),
            os.path.join(workspace, p),
            os.path.join(dnanet_root, p),
            os.path.abspath(p),
        ]
        for s in strip_names:
            cands.extend([
                os.path.join(dnanet_root, s),
                os.path.join(launch_cwd, "DNANet", s),
            ])
        # common: user passes DNANet/result/... while cwd will be DNANet
        if "result/" in p_norm or p_norm.startswith("result"):
            tail = p_norm.split("result/", 1)[-1] if "result/" in p_norm else p_norm
            cands.append(os.path.join(dnanet_root, "result", tail if not tail.startswith("result") else tail[7:]))
        seen = set()
        for c in cands:
            c_abs = os.path.abspath(c)
            if c_abs in seen:
                continue
            seen.add(c_abs)
            if os.path.exists(c_abs):
                return c_abs
        return os.path.abspath(os.path.join(launch_cwd, p))

    # Resolve all user paths BEFORE chdir into DNANet
    ckpt_path = resolve_path(args.teacher_ckpt)
    scores_path = resolve_path(args.scores_path) if args.scores_path else ""
    data_root = resolve_path(args.data_root)
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(os.path.join(launch_cwd, out_path))

    if not os.path.isfile(ckpt_path):
        # helpful listing
        result_dir = os.path.join(dnanet_root, "result")
        hint = ""
        if os.path.isdir(result_dir):
            nuaa = sorted([d for d in os.listdir(result_dir) if d.startswith("NUAA")])
            hint = f"\n[hint] DNANet/result 下 NUAA 目录: {nuaa[:8]}"
        raise SystemExit(
            f"teacher ckpt not found.\n"
            f"  given: {args.teacher_ckpt}\n"
            f"  resolved: {ckpt_path}\n"
            f"  dnanet_root: {dnanet_root}\n"
            f"  launch_cwd: {launch_cwd}"
            f"{hint}\n"
            f"可改用相对 DNANet 的路径，例如:\n"
            f"  --teacher-ckpt result/NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar"
        )

    sys.path.insert(0, dnanet_root)
    os.chdir(dnanet_root)

    from torch.utils.data import DataLoader
    from torchvision import transforms
    from model.load_param_data import load_dataset, load_param
    from model.model_DNANet import DNANet, Res_CBAM_block
    from model.utils import TestSetLoader
    from compress.adapters.dnanet_adapter import DNANetAdapter
    from compress.importance import ImportanceEstimator, l1_weight_scores

    print(f"[env] device={device}  cuda={torch.cuda.is_available()}")
    print(f"[path] dnanet_root={dnanet_root}")
    print(f"[path] teacher={ckpt_path}")
    print(f"[path] data_root={data_root}")
    if scores_path:
        print(f"[path] scores={scores_path}")
    if device.type == "cpu":
        print("[hint] 全量 11 比例 × 4 方法在 CPU 上通常要数小时；建议 --device cuda，"
              "或先用 --ratios 0,0.1,0.2,0.3 --limit 16 冒烟。")

    nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
    model = DNANet(
        num_classes=1,
        input_channels=3,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=list(nb_filter),
        deep_supervision=True,
    )
    state = load_state_dict_flexible(ckpt_path)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", list(missing)[:6])
    if unexpected:
        print("[warn] unexpected keys:", list(unexpected)[:6])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = DNANetAdapter(list(nb_filter))
    print(f"[teacher] nb_filter={list(nb_filter)}  ckpt={ckpt_path}")

    if not os.path.isdir(os.path.join(data_root, args.dataset)):
        raise SystemExit(f"dataset not found under {data_root}/{args.dataset}")
    _, test_ids, _ = load_dataset(data_root, args.dataset, args.split_method, subset_size=0, subset_seed=42)
    if args.limit > 0:
        test_ids = test_ids[: args.limit]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        TestSetLoader(
            os.path.join(data_root, args.dataset),
            img_id=test_ids,
            base_size=args.base_size,
            crop_size=args.base_size,
            transform=tf,
            suffix=".png",
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[data] {args.dataset}  n_test={len(test_ids)}  root={data_root}")

    # scores for I
    scores_I: Optional[Dict[str, torch.Tensor]] = None
    if scores_path:
        if not os.path.isfile(scores_path):
            raise SystemExit(f"scores.pt not found: {args.scores_path} -> {scores_path}")
        scores_I = load_scores(scores_path)
        print(f"[scores] loaded I from {scores_path}")
    else:
        print("[scores] no --scores-path; recomputing full I on train loader ...")
        from model.utils import TrainSetLoader
        train_ids, _, _ = load_dataset(data_root, args.dataset, args.split_method, 0, 42)
        train_set = TrainSetLoader(
            dataset_dir=os.path.join(data_root, args.dataset),
            img_id=train_ids,
            base_size=args.base_size,
            crop_size=args.base_size,
            transform=tf,
            suffix=".png",
        )
        train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        est = ImportanceEstimator(adapter, alpha=args.alpha, dilate=3, max_batches=args.score_batches)
        scores_I = est.estimate(model, train_loader, device=str(device))
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    scores_l1 = None
    if "l1" in methods:
        scores_l1 = l1_weight_scores(model, adapter, device="cpu")
        print("[scores] computed L1 weight scores")

    # baseline
    t0 = time.time()
    base_iou = eval_miou(model, loader, device, limit=0)
    print(f"[baseline] IoU={base_iou:.4f}  ({time.time()-t0:.1f}s)")

    # results[method][ratio] = iou
    results = {m: {} for m in methods}
    for m in methods:
        results[m][0.0] = base_iou

    for ratio in ratios:
        if abs(ratio) < 1e-12:
            continue
        for mode in methods:
            score_map = scores_l1 if mode == "l1" else scores_I
            assert score_map is not None
            drop = choose_drop_indices(score_map, ratio, mode, seed=args.seed)
            n_drop = sum(int(v.numel()) for v in drop.values())
            handles = attach_zero_hooks(model, adapter, drop)
            t1 = time.time()
            iou = eval_miou(model, loader, device, limit=0)
            clear_hooks(handles)
            results[mode][ratio] = iou
            print(
                f"[knock] ratio={ratio:.2f}  method={mode:<10s}  "
                f"drop_ch={n_drop:<4d}  IoU={iou:.4f}  ({time.time()-t1:.1f}s)"
            )

    # write CSV（out_path 已在 chdir 前解析为绝对路径）
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # column order fixed for MATLAB
    col_map = {
        "topk_I": "topk_I",
        "bottomk_I": "bottomk_I",
        "random": "random",
        "l1": "l1",
    }
    header = ["remove_ratio"] + [col_map[m] for m in ["topk_I", "bottomk_I", "random", "l1"] if m in results]
    # always write all four columns if present in methods; fill empty if missing
    header = ["remove_ratio", "topk_I", "bottomk_I", "random", "l1"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ratio in ratios:
            row = [f"{ratio:.2f}"]
            for m in ["topk_I", "bottomk_I", "random", "l1"]:
                if m in results and ratio in results[m]:
                    row.append(f"{results[m][ratio]:.4f}")
                elif m in results and abs(ratio) < 1e-12:
                    row.append(f"{base_iou:.4f}")
                else:
                    row.append("")
            w.writerow(row)

    print(f"[save] {out_path}")
    print("下一步 MATLAB：")
    print("  cd('.../images/Matlab/敲除校准曲线')")
    print("  plot_knockout_curves('schematic', false)")


if __name__ == "__main__":
    main()
