"""
测量 MSHNet 教师/学生 Params / FLOPs（MACs×2）。

  python measure_mshnet_complexity.py \\
    --teacher_ckpt ../MSHNet-main/weight/MSHNet-NUAA-SIRST-*/weight.pkl \\
    --student_ckpt ../MSHNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar \\
    --input_size 256 --out outputs/mshnet/complexity_nuaa.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MSH = os.path.join(_ROOT, "MSHNet-main")
if _MSH not in sys.path:
    sys.path.insert(0, _MSH)

from model.MSHNet import MSHNet, DEFAULT_CHANNELS  # noqa: E402


class SingleOutputWrapper(nn.Module):
    def __init__(self, model, warm=True):
        super().__init__()
        self.model = model
        self.warm = warm

    def forward(self, x):
        out = self.model(x, self.warm)
        if isinstance(out, (list, tuple)):
            if len(out) == 2 and isinstance(out[0], (list, tuple)):
                return out[1]
            return out[-1]
        return out


def load_build(ckpt_path: str, in_ch: int = 3):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and any(torch.is_tensor(v) for v in ckpt.values()) and not any(
        k in ckpt for k in ("state_dict", "net", "model")
    ):
        state, meta = ckpt, {}
    else:
        state = None
        for key in ("state_dict", "net", "model"):
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            raise ValueError(f"no state_dict in {ckpt_path}")
        meta = ckpt if isinstance(ckpt, dict) else {}
    state = {k.replace("module.", ""): v for k, v in state.items()}
    channels = meta.get("param_channels") or meta.get("channels") or list(DEFAULT_CHANNELS)
    model = MSHNet(in_ch, channels=list(channels))
    model.load_state_dict(state, strict=False)
    model.eval()
    return SingleOutputWrapper(model, warm=True), {
        "best_iou": meta.get("best_iou", ""),
        "channels": list(model.param_channels),
    }


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def macs_with_thop(model, input_size, in_ch):
    from thop import profile

    x = torch.randn(1, in_ch, input_size, input_size)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return float(macs)


def macs_manual(model, input_size, in_ch):
    total = [0.0]

    def conv_hook(m, inp, out):
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
    model, meta = load_build(ckpt, in_ch=in_ch)
    params = count_params(model)
    method = "thop"
    try:
        macs = macs_with_thop(model, input_size, in_ch)
    except Exception as e:
        print(f"[warn] thop failed ({e}), fallback manual Conv MACs")
        macs = macs_manual(model, input_size, in_ch)
        method = "manual_conv"
    flops = macs * 2.0
    row = {
        "tag": tag,
        "ckpt": os.path.abspath(ckpt),
        "input_size": input_size,
        "params": params,
        "params_M": round(params / 1e6, 3),
        "macs": int(macs),
        "macs_G": round(macs / 1e9, 3),
        "flops": int(flops),
        "flops_G": round(flops / 1e9, 3),
        "method": method,
        "best_iou": meta.get("best_iou", ""),
        "channels": ",".join(str(x) for x in meta["channels"]),
    }
    print(
        f"[{tag}] Params={row['params_M']}M  MACs={row['macs_G']}G  "
        f"FLOPs={row['flops_G']}G  ({method})  ch={row['channels']}"
    )
    return row


def main():
    p = argparse.ArgumentParser(description="Measure MSHNet Params/FLOPs")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--out", default="outputs/mshnet/complexity_nuaa.csv")
    args = p.parse_args()

    for path in (args.teacher_ckpt, args.student_ckpt):
        if not os.path.isfile(path):
            raise SystemExit(f"[error] missing: {path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    rows = [
        measure("teacher", args.teacher_ckpt, args.input_size, args.in_channels),
        measure("student", args.student_ckpt, args.input_size, args.in_channels),
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {os.path.abspath(args.out)}")

    t, s = rows[0], rows[1]

    def pct(a, b):
        return (b - a) / a * 100.0 if a else 0.0

    print("--- delta (student vs teacher) ---")
    print(f"Params: {pct(t['params'], s['params']):+.1f}%")
    print(f"FLOPs:  {pct(t['flops'], s['flops']):+.1f}%")


if __name__ == "__main__":
    main()
