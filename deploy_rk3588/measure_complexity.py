"""
测量 DNANet 教师/学生 参数量与 FLOPs（MACs×2 近似）。

优先用 thop；没有则用手算 Conv/Linear 的 MACs。
在 AutoDL（有 torch）上运行:

  python measure_complexity.py \
    --teacher_ckpt ../Infrared-Small-Target-Detection-master/result/NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar \
    --student_nb_filter 13,22,38,64,102 \
    --input_size 256 \
    --out outputs/complexity_nuaa.csv

若不传 teacher_ckpt，则按默认 nb_filter=[16,32,64,128,256] 建教师结构计 FLOPs。
学生可用 --student_ckpt（含 nb_filter）或 --student_nb_filter。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DNANET = os.path.join(_ROOT, "Infrared-Small-Target-Detection-master")
if _DNANET not in sys.path:
    sys.path.insert(0, _DNANET)

from model.load_param_data import load_param  # noqa: E402
from model.model_DNANet import DNANet, Res_CBAM_block  # noqa: E402


class SingleOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out


def build_dnanet(nb_filter, num_blocks, deep_sup=True, in_ch=3):
    return DNANet(
        num_classes=1,
        input_channels=in_ch,
        block=Res_CBAM_block,
        num_blocks=list(num_blocks),
        nb_filter=list(nb_filter),
        deep_supervision=deep_sup,
    )


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def macs_with_thop(model, input_size, in_ch):
    from thop import profile

    x = torch.randn(1, in_ch, input_size, input_size)
    macs, params = profile(model, inputs=(x,), verbose=False)
    return float(macs), float(params)


def macs_manual_hooks(model, input_size, in_ch):
    """无 thop 时：hook Conv2d/Linear 累计 MACs。"""
    total_macs = [0.0]

    def conv_hook(m, inp, out):
        # out: N,C,H,W ; weight: C_out, C_in/groups, kH, kW
        _, _, h, w = out.shape
        cout, cin_g, kh, kw = m.weight.shape
        groups = m.groups
        total_macs[0] += h * w * cout * (cin_g * kh * kw) * groups

    def linear_hook(m, inp, out):
        # weight: out, in
        total_macs[0] += m.weight.shape[0] * m.weight.shape[1]

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(1, in_ch, input_size, input_size))
    for h in handles:
        h.remove()
    return total_macs[0]


def measure_one(name, model, input_size, in_ch):
    model = SingleOutputWrapper(model).eval()
    n_params, n_train = count_params(model)
    try:
        macs, _ = macs_with_thop(model, input_size, in_ch)
        method = "thop"
    except Exception as e:
        print(f"[{name}] thop unavailable ({e}), fallback manual hooks")
        macs = macs_manual_hooks(model, input_size, in_ch)
        method = "manual_conv_linear"

    flops = macs * 2.0  # 习惯上 FLOPs ≈ 2×MACs（乘加各计一次）
    row = {
        "name": name,
        "params": int(n_params),
        "params_M": round(n_params / 1e6, 4),
        "trainable_params": int(n_train),
        "macs": int(macs),
        "macs_G": round(macs / 1e9, 4),
        "flops": int(flops),
        "flops_G": round(flops / 1e9, 4),
        "input": f"1x{in_ch}x{input_size}x{input_size}",
        "method": method,
    }
    return row


def load_nb_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "nb_filter" in ckpt:
        return list(ckpt["nb_filter"])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", type=str, default="")
    p.add_argument("--student_ckpt", type=str, default="")
    p.add_argument("--teacher_nb_filter", type=str, default="16,32,64,128,256")
    p.add_argument("--student_nb_filter", type=str, default="13,22,38,64,102")
    p.add_argument("--channel_size", type=str, default="three")
    p.add_argument("--backbone", type=str, default="resnet_18")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--out", type=str, default="outputs/complexity_nuaa.csv")
    args = p.parse_args()

    _, num_blocks = load_param(args.channel_size, args.backbone)

    t_nb = [int(x) for x in args.teacher_nb_filter.split(",")]
    if args.teacher_ckpt and os.path.isfile(args.teacher_ckpt):
        nb = load_nb_from_ckpt(args.teacher_ckpt)
        if nb:
            t_nb = nb
        print(f"[teacher] ckpt={args.teacher_ckpt} nb_filter={t_nb}")
    else:
        print(f"[teacher] structural nb_filter={t_nb}")

    s_nb = [int(x) for x in args.student_nb_filter.split(",")]
    if args.student_ckpt and os.path.isfile(args.student_ckpt):
        nb = load_nb_from_ckpt(args.student_ckpt)
        if nb:
            s_nb = nb
        print(f"[student] ckpt={args.student_ckpt} nb_filter={s_nb}")
    else:
        print(f"[student] structural nb_filter={s_nb}")

    teacher = build_dnanet(t_nb, num_blocks)
    student = build_dnanet(s_nb, num_blocks)

    if args.teacher_ckpt and os.path.isfile(args.teacher_ckpt):
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        teacher.load_state_dict(state, strict=False)

    if args.student_ckpt and os.path.isfile(args.student_ckpt):
        ckpt = torch.load(args.student_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        student.load_state_dict(state, strict=False)

    rows = [
        measure_one("teacher", teacher, args.input_size, args.in_channels),
        measure_one("student", student, args.input_size, args.in_channels),
    ]

    # 相对变化
    t, s = rows[0], rows[1]
    delta = {
        "name": "delta_student_vs_teacher",
        "params": s["params"] - t["params"],
        "params_M": round(s["params_M"] - t["params_M"], 4),
        "trainable_params": s["trainable_params"] - t["trainable_params"],
        "macs": s["macs"] - t["macs"],
        "macs_G": round(s["macs_G"] - t["macs_G"], 4),
        "flops": s["flops"] - t["flops"],
        "flops_G": round(s["flops_G"] - t["flops_G"], 4),
        "input": t["input"],
        "method": f"params%={100*(s['params']/t['params']-1):.2f}; flops%={100*(s['flops']/t['flops']-1):.2f}",
    }
    rows.append(delta)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("======== Complexity ========")
    for r in rows:
        print(
            f"{r['name']}: params={r['params_M']}M  MACs={r['macs_G']}G  "
            f"FLOPs={r['flops_G']}G  ({r['method']})"
        )
    print(f"[save] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
