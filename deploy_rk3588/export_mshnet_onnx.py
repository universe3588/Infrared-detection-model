"""
PC/AutoDL: 将 MSHNet 教师/学生权重导出为单输出 ONNX，供 RK3588 转换。

用法（在 deploy_rk3588 目录）:
  python export_mshnet_onnx.py \\
    --ckpt ../MSHNet-main/weight/MSHNet-NUAA-SIRST-*/weight.pkl \\
    --out outputs/mshnet/teacher_nuaa.onnx --input_size 256

  python export_mshnet_onnx.py \\
    --ckpt ../MSHNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar \\
    --out outputs/mshnet/student_nuaa.onnx --input_size 256

学生 ckpt 须含 param_channels / channels（compress_train 会写入）。
"""

from __future__ import annotations

import argparse
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
    """warm_flag=True 取融合最终输出，保证 ONNX/RKNN 单输出。"""

    def __init__(self, model: nn.Module, warm: bool = True):
        super().__init__()
        self.model = model
        self.warm = warm

    def forward(self, x):
        out = self.model(x, self.warm)
        if isinstance(out, (list, tuple)):
            # (sides, final) → final
            if len(out) == 2 and isinstance(out[0], (list, tuple)):
                return out[1]
            return out[-1]
        return out


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt type: {type(ckpt)}")
    # weight.pkl 可能是纯 state_dict；checkpoint/student 含嵌套
    if any(torch.is_tensor(v) for v in ckpt.values()) and not any(
        k in ckpt for k in ("state_dict", "net", "model")
    ):
        state = ckpt
        meta = {}
    else:
        state = None
        for key in ("state_dict", "net", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            raise ValueError(f"no state_dict in {path}")
        meta = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    channels = meta.get("param_channels") or meta.get("channels")
    return state, channels, meta


def main():
    p = argparse.ArgumentParser(description="Export MSHNet to ONNX for RK3588")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/mshnet/mshnet.onnx")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--opset", type=int, default=11)
    p.add_argument("--warm", action="store_true", default=True,
                   help="导出多尺度融合最终头（默认开）")
    p.add_argument("--no_warm", action="store_true", help="仅 output_0 单头")
    args = p.parse_args()
    warm = not args.no_warm

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"[error] ckpt not found: {args.ckpt}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    state, channels, meta = load_ckpt(args.ckpt)
    if channels is None:
        print("[info] ckpt 无 channels，按教师默认 [16,32,64,128,256] 构建")
        channels = list(DEFAULT_CHANNELS)
    else:
        channels = list(channels)
        print(f"[info] channels={channels}")

    model = MSHNet(args.in_channels, channels=channels)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    if missing and len(missing) > len(state) * 0.2:
        raise SystemExit("[error] 过多 missing keys，通道配置可能与权重不匹配")

    wrapped = SingleOutputWrapper(model, warm=warm).eval()
    dummy = torch.randn(1, args.in_channels, args.input_size, args.input_size)
    with torch.no_grad():
        y = wrapped(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] channels={list(model.param_channels)} warm={warm}")
    print(f"[export] params={n_params / 1e6:.3f}M")
    print(f"[export] input={tuple(dummy.shape)} output={tuple(y.shape)}")
    if isinstance(meta, dict) and "best_iou" in meta:
        print(f"[export] ckpt best_iou={meta['best_iou']}")

    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["mask_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[save] ONNX -> {os.path.abspath(args.out)}")
    print(f"[size] {os.path.getsize(args.out) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
