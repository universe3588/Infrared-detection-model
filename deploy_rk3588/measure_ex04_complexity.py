#!/usr/bin/env python3
"""thop 统计 L2SK / RepISD Params & FLOPs（@256），写入 outputs/ex04/complexity.csv"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from thop import profile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def measure_l2sk(ckpt: str, size: int = 256):
    sys.path.insert(0, os.path.join(_ROOT, "Lightweight model comparison", "L2SKNet"))
    from net import Net

    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net = Net("L2SKNet_UNet").eval()
    net.load_state_dict(state, strict=True)
    x = torch.randn(1, 1, size, size)
    flops, params = profile(net, inputs=(x,), verbose=False)
    return params / 1e6, flops / 1e9, "1x1x{0}x{0}".format(size)


def measure_repisd(ckpt: str, size: int = 256):
    sys.path.insert(0, os.path.join(_ROOT, "Lightweight model comparison", "RepISD-Net"))
    from convert_tool.convert_CDC import config_model
    from model.RepISD import RepISD

    cdc = config_model("cdc")
    net = RepISD(cdc[0], bins=(1, 2, 3, 6), dropout=0.1, classes=2, convert=True, deploy=True)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    net.eval()
    x = torch.randn(1, 3, size, size)
    flops, params = profile(net, inputs=(x,), verbose=False)
    return params / 1e6, flops / 1e9, "1x3x{0}x{0}".format(size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--l2sk-ckpt", default="../Lightweight model comparison/L2SKNet/log/NUAA-SIRST/L2SKNet_UNet/best_miou.pth.tar")
    p.add_argument("--repisd-ckpt", default="../Lightweight model comparison/RepISD-Net/result/NUAA/deploy.pth.tar")
    p.add_argument("--out", default="outputs/ex04/complexity.csv")
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    rows = []
    # DNA student reference from paper table
    rows.append(
        {
            "model": "DNANet_student",
            "params_M": 1.13,
            "flops_G": 12.92,
            "input": "1x3x256x256",
            "note": "from tab:resource (ONNX≈2xMACs)",
        }
    )
    if os.path.isfile(args.l2sk_ckpt):
        pm, fl, inp = measure_l2sk(args.l2sk_ckpt, args.size)
        rows.append({"model": "L2SKNet_UNet", "params_M": round(pm, 4), "flops_G": round(fl, 4), "input": inp, "note": "thop"})
        print(f"L2SK  params={pm:.4f}M  FLOPs={fl:.4f}G")
    else:
        print("[warn] skip L2SK, missing", args.l2sk_ckpt)

    if os.path.isfile(args.repisd_ckpt):
        pm, fl, inp = measure_repisd(args.repisd_ckpt, args.size)
        rows.append({"model": "RepISD", "params_M": round(pm, 4), "flops_G": round(fl, 4), "input": inp, "note": "thop"})
        print(f"RepISD params={pm:.4f}M  FLOPs={fl:.4f}G")
    else:
        print("[warn] skip RepISD, missing", args.repisd_ckpt)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "params_M", "flops_G", "input", "note"])
        w.writeheader()
        w.writerows(rows)
    print("[save]", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
