"""
PC/AutoDL: SCTransNet 教师/学生 → 单输出 ONNX（mode=test，仅最终头）。

输入默认 **1 通道灰度**（与训练一致）；板端对照仍用 256×256。

用法（在 deploy_rk3588 目录）:
  python export_sctrans_onnx.py \\
    --ckpt ../SCTransNet-main/log/NUAA-SIRST/SCTransNet_781_best.pth.tar \\
    --out outputs/sctrans/teacher_nuaa.onnx

  python export_sctrans_onnx.py \\
    --ckpt ../SCTransNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar \\
    --out outputs/sctrans/student_nuaa.onnx
"""

from __future__ import annotations

import argparse
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
    """强制 test 模式，保证单输出 sigmoid mask。"""

    def __init__(self, model: SCTransNet):
        super().__init__()
        self.model = model
        self.model.mode = "test"

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt type: {type(ckpt)}")
    state = None
    for key in ("state_dict", "net", "model", "student", "teacher"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None and any(torch.is_tensor(v) for v in ckpt.values()):
        state, meta = ckpt, {}
    else:
        meta = ckpt
    if state is None:
        raise ValueError(f"no state_dict in {path}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    if isinstance(state, dict):
        # 偶发嵌套 model.
        if any(k.startswith("model.") for k in state):
            state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    channels = None
    if isinstance(meta, dict):
        channels = meta.get("channel_num") or meta.get("channels")
    return state, channels, meta if isinstance(meta, dict) else {}


def build_model(channels, img_size: int, n_channels: int = 1) -> SCTransNet:
    cfg = get_SCTrans_config()
    model = SCTransNet(
        cfg,
        n_channels=n_channels,
        n_classes=1,
        img_size=img_size,
        mode="test",
        deepsuper=True,
        channel_num=list(channels),
    )
    return model


def main():
    p = argparse.ArgumentParser(description="Export SCTransNet to ONNX for RK3588")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/sctrans/sctrans.onnx")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=1, help="训练默认灰度 1 通道")
    p.add_argument("--opset", type=int, default=11)
    p.add_argument(
        "--channels", type=str, default="",
        help="可选覆盖，如 30,54,96,166；默认读 ckpt 或教师 [32,64,128,256]",
    )
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"[error] ckpt not found: {args.ckpt}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    state, channels, meta = load_ckpt(args.ckpt)
    if args.channels.strip():
        channels = [int(x) for x in args.channels.split(",")]
    elif channels is None:
        channels = list(TEACHER_CHANNELS)
        print(f"[info] ckpt 无 channel_num，按教师默认 {channels}")
    else:
        channels = [int(x) for x in channels]
    print(f"[info] channels={channels}")

    model = build_model(channels, args.input_size, n_channels=args.in_channels)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    if missing and len(missing) > max(8, int(len(state) * 0.2)):
        raise SystemExit("[error] 过多 missing keys，通道配置可能与权重不匹配")

    wrapped = TestModeWrapper(model).eval()
    dummy = torch.randn(1, args.in_channels, args.input_size, args.input_size)
    with torch.no_grad():
        y = wrapped(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] params={n_params / 1e6:.3f}M  KV_size={sum(channels)}")
    print(f"[export] input={tuple(dummy.shape)} output={tuple(y.shape)}")
    if "best_iou" in meta:
        print(f"[export] ckpt best_iou={meta['best_iou']}")

    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[save] ONNX -> {os.path.abspath(args.out)}")
    print(f"[size] {os.path.getsize(args.out) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
