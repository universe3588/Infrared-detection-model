#!/usr/bin/env python3
"""
L2SKNet-UNet → 单输出 ONNX（1 通道灰度，默认 256×256）。

与 SCTrans 板端协议一致：单通道 @256；论文表注需写明 DNA 学生为三通道。

用法（deploy_rk3588 目录）:
  python export_l2sk_onnx.py \\
    --ckpt \"../Lightweight model comparison/L2SKNet/log/NUAA-SIRST/L2SKNet_UNet/best_miou.pth.tar\" \\
    --out outputs/ex04/l2sk_unet_nuaa.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_L2SK = os.path.join(_ROOT, "Lightweight model comparison", "L2SKNet")
if _L2SK not in sys.path:
    sys.path.insert(0, _L2SK)

from net import Net  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="outputs/ex04/l2sk_unet.onnx")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--model_name", type=str, default="L2SKNet_UNet")
    p.add_argument("--opset", type=int, default=11)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    net = Net(model_name=args.model_name)
    net.load_state_dict(state, strict=True)
    net.eval()

    dummy = torch.randn(1, 1, args.input_size, args.input_size)
    torch.onnx.export(
        net,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    n = sum(x.numel() for x in net.parameters())
    print(f"[save] {os.path.abspath(args.out)}")
    print(f"[info] params={n/1e6:.4f}M  input=1x1x{args.input_size}x{args.input_size}")


if __name__ == "__main__":
    main()
