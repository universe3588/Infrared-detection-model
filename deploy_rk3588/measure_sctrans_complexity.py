"""
测量 SCTransNet 教师/学生 Params / FLOPs（MACs×2），@256，默认 1 通道。

  python measure_sctrans_complexity.py \\
    --teacher_ckpt ../SCTransNet-main/log/NUAA-SIRST/SCTransNet_*_best.pth.tar \\
    --student_ckpt ../SCTransNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar \\
    --out outputs/sctrans/complexity_nuaa.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCT = os.path.join(_ROOT, "SCTransNet-main")
if _SCT not in sys.path:
    sys.path.insert(0, _SCT)

from compress.adapters.sctrans_adapter import TEACHER_CHANNELS  # noqa: E402
from model.Config import get_SCTrans_config  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402


class TestModeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.mode = "test"

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out


def load_build(ckpt_path: str, img_size: int, in_ch: int = 1):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    meta = {}
    if isinstance(ckpt, dict):
        for key in ("state_dict", "net", "model", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                meta = ckpt
                break
        if state is None and any(torch.is_tensor(v) for v in ckpt.values()):
            state, meta = ckpt, {}
    if state is None:
        raise ValueError(f"no state_dict in {ckpt_path}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    channels = meta.get("channel_num") or meta.get("channels") or list(TEACHER_CHANNELS)
    channels = [int(x) for x in channels]
    cfg = get_SCTrans_config()
    model = SCTransNet(
        cfg, n_channels=in_ch, img_size=img_size, mode="test",
        deepsuper=True, channel_num=channels,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return TestModeWrapper(model), {
        "best_iou": meta.get("best_iou", ""),
        "channels": channels,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    }


def macs_with_thop(model, input_size, in_ch):
    from thop import profile
    x = torch.randn(1, in_ch, input_size, input_size)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return float(macs)


def macs_manual(model, input_size, in_ch):
    total = [0.0]

    def conv_hook(m, inp, out):
        if not hasattr(out, "shape") or len(out.shape) != 4:
            return
        _, _, h, w = out.shape
        cout, cin_g, kh, kw = m.weight.shape
        total[0] += h * w * cout * (cin_g * kh * kw) * m.groups

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
    with torch.no_grad():
        model(torch.randn(1, in_ch, input_size, input_size))
    for h in handles:
        h.remove()
    return total[0]


def measure(tag, ckpt, input_size, in_ch):
    model, meta = load_build(ckpt, input_size, in_ch)
    params = sum(p.numel() for p in model.parameters())
    method = "thop"
    try:
        macs = macs_with_thop(model, input_size, in_ch)
    except Exception as e:
        print(f"[warn] thop failed ({e}); fallback conv-hook MACs")
        macs = macs_manual(model, input_size, in_ch)
        method = "conv_hook"
    flops = 2.0 * macs
    row = {
        "tag": tag,
        "ckpt": ckpt,
        "params_M": round(params / 1e6, 3),
        "flops_G": round(flops / 1e9, 3),
        "macs_G": round(macs / 1e9, 3),
        "method": method,
        "channels": str(meta["channels"]),
        "best_iou": meta.get("best_iou", ""),
        "in_channels": in_ch,
        "input_size": input_size,
    }
    print(
        f"[{tag}] params={row['params_M']}M  FLOPs={row['flops_G']}G  "
        f"({method})  ch={row['channels']}"
    )
    return row


def pct(a, b):
    if a == 0:
        return float("nan")
    return (b - a) / a * 100.0


def main():
    p = argparse.ArgumentParser(description="Measure SCTransNet Params/FLOPs")
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--out", type=str, default="outputs/sctrans/complexity_nuaa.csv")
    args = p.parse_args()

    rows = [
        measure("teacher", args.teacher_ckpt, args.input_size, args.in_channels),
        measure("student", args.student_ckpt, args.input_size, args.in_channels),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    t, s = rows[0], rows[1]
    print("-------- delta (student vs teacher) --------")
    print(f"Params: {pct(t['params_M'], s['params_M']):+.1f}%")
    print(f"FLOPs:  {pct(t['flops_G'], s['flops_G']):+.1f}%")
    print(f"[save] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
