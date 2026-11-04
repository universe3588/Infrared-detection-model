"""
PC/AutoDL: 将 UIU-Net 教师/学生权重导出为单输出 ONNX，供 RK3588 转换。

用法（在 deploy_rk3588 目录）:
  python export_uiunet_onnx.py \
    --ckpt ../IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd/NUAA-SIRST_*/uiunet_best.pth.tar \
    --out outputs/uiunet/teacher_nuaa.onnx --input_size 256

  python export_uiunet_onnx.py \
    --ckpt ../IEEE_TIP_UIU-Net-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar \
    --out outputs/uiunet/student_nuaa.onnx --input_size 256

学生 ckpt 必须含 enc_outs / dec_outs / mids_enc / mids_dec（compress_train 会写入）。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_UIU = os.path.join(_ROOT, "IEEE_TIP_UIU-Net-main")
if _UIU not in sys.path:
    sys.path.insert(0, _UIU)

from model.uiunet import (  # noqa: E402
    UIUNET,
    DEFAULT_ENC_OUTS,
    DEFAULT_DEC_OUTS,
    DEFAULT_MIDS_ENC,
    DEFAULT_MIDS_DEC,
)


class SingleOutputWrapper(nn.Module):
    """多侧输出时只取最终 d0，保证 ONNX/RKNN 单输出。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt type: {type(ckpt)}")
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"no state_dict in {path}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    enc = ckpt.get("enc_outs", None)
    dec = ckpt.get("dec_outs", None)
    me = ckpt.get("mids_enc", None)
    md = ckpt.get("mids_dec", None)
    return state, enc, dec, me, md, ckpt


def build_model(enc, dec, me, md, in_ch=3, out_ch=1):
    return UIUNET(
        in_ch=in_ch,
        out_ch=out_ch,
        enc_outs=list(enc) if enc is not None else list(DEFAULT_ENC_OUTS),
        dec_outs=list(dec) if dec is not None else list(DEFAULT_DEC_OUTS),
        mids_enc=list(me) if me is not None else list(DEFAULT_MIDS_ENC),
        mids_dec=list(md) if md is not None else list(DEFAULT_MIDS_DEC),
    )


def main():
    p = argparse.ArgumentParser(description="Export UIU-Net to ONNX for RK3588")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/uiunet/uiunet.onnx")
    p.add_argument("--input_size", type=int, default=256,
                   help="与 DNANet 板端对比默认 256；官方评测常用 320")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--opset", type=int, default=11)
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"[error] ckpt not found: {args.ckpt}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    state, enc, dec, me, md, meta = load_ckpt(args.ckpt)
    if enc is None:
        print("[info] ckpt 无宽度字段，按教师默认通道构建（适用于 uiunet_best 教师）")
    else:
        print(f"[info] student/custom widths enc={list(enc)} dec={list(dec)}")

    model = build_model(enc, dec, me, md, in_ch=args.in_channels)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    if missing and len(missing) > len(state) * 0.2:
        raise SystemExit("[error] 过多 missing keys，通道配置可能与权重不匹配")

    wrapped = SingleOutputWrapper(model).eval()
    dummy = torch.randn(1, args.in_channels, args.input_size, args.input_size)
    with torch.no_grad():
        y = wrapped(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] enc={list(model.enc_outs)} dec={list(model.dec_outs)}")
    print(f"[export] params={n_params / 1e6:.3f}M")
    print(f"[export] input={tuple(dummy.shape)} output={tuple(y.shape)}")
    if "best_iou" in meta:
        print(f"[export] ckpt best_iou={meta['best_iou']}")

    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["mask_prob"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[save] ONNX -> {os.path.abspath(args.out)}")
    print(f"[size] {os.path.getsize(args.out) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
