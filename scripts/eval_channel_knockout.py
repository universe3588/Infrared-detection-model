#!/usr/bin/env python3
"""
DNA-Net 通道敲除校准（Ex08 实测版）：按重要性 Top-k / Bottom-k / random / l1
逐步置零通道，评测 IoU，写出与 MATLAB 兼容的 CSV。

示例（在 Infrared-Small-Target-Detection-master 下）：
  python ../scripts/eval_channel_knockout.py \\
    --teacher_ckpt result/NUAA-SIRST_DNANet_*_wDS/mIoU__DNANet_*_epoch.pth.tar \\
    --scores_path compress_runs/.../scores.pt \\
    --dataset NUAA-SIRST \\
    --out ../面向边缘部署的红外弱小目标检测可迁移压缩框架/images/Matlab/data/knockout_iou_nuaa.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="Infrared-Small-Target-Detection-master")
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--scores_path", type=str, required=True, help="scores.pt from P1")
    p.add_argument("--dataset", type=str, default="NUAA-SIRST")
    p.add_argument("--root", type=str, default="../RDANet-main/datasets")
    p.add_argument("--ratios", type=str, default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--gpus", type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)

    from torch.utils.data import DataLoader
    from torchvision import transforms
    from model.load_param_data import load_dataset, load_param
    from model.model_DNANet import DNANet, Res_CBAM_block
    from model.utils import TestSetLoader
    from model.metric import mIoU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_filter, num_blocks = load_param("three", "resnet_18")
    model = DNANet(
        num_classes=1, input_channels=3, block=Res_CBAM_block,
        num_blocks=num_blocks, nb_filter=list(nb_filter), deep_supervision=True,
    )
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    payload = torch.load(args.scores_path, map_location="cpu")
    scores = payload["scores"] if isinstance(payload, dict) and "scores" in payload else payload

    # 简化：对每 stage 的 score 向量做全局排序需要 adapter；此处按 stage 内相对排序聚合
    # 期望 scores: Dict[stage_id, Tensor[C]]
    if not isinstance(scores, dict):
        raise SystemExit("scores.pt must contain dict of per-stage score vectors")

    _, val_ids, _ = load_dataset(args.root, args.dataset, "50_50", subset_size=0, subset_seed=42)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    loader = DataLoader(
        TestSetLoader(
            os.path.join(args.root, args.dataset), img_id=val_ids,
            base_size=256, crop_size=256, transform=tf, suffix=".png",
        ),
        batch_size=1, shuffle=False, num_workers=0,
    )

    ratios = [float(x) for x in args.ratios.split(",")]
    rows = []
    print("[warn] 本脚本为脚手架：完整 Top-k 置零需绑定 DNA Adapter 的通道切片钩子。")
    print("[warn] 当前先写出表头与 baseline（ratio=0）教师 IoU，便于 MATLAB 管线打通。")
    print("[warn] 请在服务器上按 compress/importance 的 stage keep 逻辑补全 knock 循环。")

    # baseline
    metric = mIoU(1)
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            pred = pred[-1] if isinstance(pred, (list, tuple)) else pred
            metric.update(pred, labels)
    _, base_iou = metric.get()
    base_iou = float(base_iou)
    print(f"[baseline] teacher IoU={base_iou:.4f}")

    # 占位：仅 ratio=0 有实测；其余留空让用户补全 / 或后续扩展
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["remove_ratio", "topk_I", "bottomk_I", "random", "l1"])
        for r in ratios:
            if abs(r) < 1e-12:
                w.writerow([f"{r:.2f}", f"{base_iou:.4f}", f"{base_iou:.4f}", f"{base_iou:.4f}", f"{base_iou:.4f}"])
            else:
                w.writerow([f"{r:.2f}", "", "", "", ""])
    print(f"[save] {args.out}")
    print("请实现通道置零后重跑填满 CSV，再 MATLAB: plot_knockout_curves('schematic', false)")


if __name__ == "__main__":
    main()
