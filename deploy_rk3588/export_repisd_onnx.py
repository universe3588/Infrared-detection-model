#!/usr/bin/env python3
"""
RepISD-Net (deploy) → ONNX。默认导出前景通道（1 通道 mask），输入 3×256×256。

用法:
  python export_repisd_onnx.py \\
    --ckpt \"../Lightweight model comparison/RepISD-Net/result/NUAA/deploy.pth.tar\" \\
    --out outputs/ex04/repisd_nuaa.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REP = os.path.join(_ROOT, "Lightweight model comparison", "RepISD-Net")
if _REP not in sys.path:
    sys.path.insert(0, _REP)

from convert_tool.convert_CDC import config_model  # noqa: E402
from model.RepISD import RepISD  # noqa: E402


class ForegroundHead(nn.Module):
    """取通道 1（前景），与 test.py 中 preds[:,1:,:,:] 一致。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y[:, 1:2, :, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="outputs/ex04/repisd.onnx")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--opset", type=int, default=11)
    p.add_argument("--config", type=str, default="cdc")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cdc = config_model(args.config)
    net = RepISD(cdc[0], bins=(1, 2, 3, 6), dropout=0.1, classes=2, convert=True, deploy=True)
    try:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)
    wrapped = ForegroundHead(net).eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    n = sum(x.numel() for x in net.parameters())
    print(f"[save] {os.path.abspath(args.out)}")
    print(f"[info] params={n/1e6:.4f}M  input=1x3x{args.input_size}x{args.input_size}")


if __name__ == "__main__":
    main()
