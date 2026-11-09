from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from compress.adapters.uiunet_adapter import (
    ENC_STAGE_IDS,
    DEC_STAGE_IDS,
    TEACHER_STAGE_WIDTHS,
)
from model.uiunet import (
    DEFAULT_ENC_OUTS,
    DEFAULT_DEC_OUTS,
    DEFAULT_MIDS_ENC,
    DEFAULT_MIDS_DEC,
    REBNCONV,
    UIUNET,
)
from model.fusion import AsymBiChaFuseReduce


def save_keep_masks(path: str, keep_idx: Dict[int, torch.Tensor], meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"keep_idx": {int(k): v.cpu() for k, v in keep_idx.items()}, "meta": meta}
    torch.save(payload, path)
    print(f"[save] keep masks -> {path}")


def load_keep_masks(path: str):
    payload = torch.load(path, map_location="cpu")
    keep_idx = {int(k): v for k, v in payload["keep_idx"].items()}
    return keep_idx, payload.get("meta", {})


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def shrink_idx(n_src: int, n_dst: int) -> torch.Tensor:
    if n_dst >= n_src:
        return torch.arange(n_src)
    return torch.linspace(0, n_src - 1, steps=n_dst).round().long().clamp(0, n_src - 1)


def _slice_conv(src: nn.Conv2d, dst: nn.Conv2d,
                keep_out: Optional[torch.Tensor], keep_in: Optional[torch.Tensor]):
    with torch.no_grad():
        w = src.weight.data
        if keep_out is not None:
            w = w[keep_out]
        if keep_in is not None:
            w = w[:, keep_in]
        if w.shape != dst.weight.shape:
            raise RuntimeError(
                f"conv slice shape mismatch: got {tuple(w.shape)} vs dst {tuple(dst.weight.shape)}"
            )
        dst.weight.data.copy_(w)
        if src.bias is not None and dst.bias is not None:
            b = src.bias.data
            if keep_out is not None:
                b = b[keep_out]
            dst.bias.data.copy_(b)


def _slice_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, keep_out: torch.Tensor):
    with torch.no_grad():
        dst.weight.data.copy_(src.weight.data[keep_out])
        dst.bias.data.copy_(src.bias.data[keep_out])
        dst.running_mean.copy_(src.running_mean.data[keep_out])
        dst.running_var.copy_(src.running_var.data[keep_out])
        dst.num_batches_tracked.copy_(src.num_batches_tracked)


def transfer_rebn(src: REBNCONV, dst: REBNCONV,
                  keep_in: torch.Tensor, keep_out: torch.Tensor):
    _slice_conv(src.conv_s1, dst.conv_s1, keep_out, keep_in)
    _slice_bn(src.bn_s1, dst.bn_s1, keep_out)


def scale_mid(mid_t: int, out_t: int, out_s: int, min_keep: int = 4) -> int:
    return max(min_keep, int(round(mid_t / float(out_t) * out_s)))


def student_channel_config(
    keep_idx: Dict[int, torch.Tensor],
    min_keep: int = 4,
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """Return enc_outs, dec_outs, mids_enc, mids_dec, stage_widths_student."""
    sw = [int(keep_idx[s].numel()) for s in range(4)]
    enc = [sw[i] for i in ENC_STAGE_IDS]
    dec = [sw[i] for i in DEC_STAGE_IDS]
    me, md = [], []
    for i, out_t in enumerate(DEFAULT_ENC_OUTS):
        sid = ENC_STAGE_IDS[i]
        me.append(scale_mid(DEFAULT_MIDS_ENC[i], out_t, enc[i], min_keep))
    for i, out_t in enumerate(DEFAULT_DEC_OUTS):
        sid = DEC_STAGE_IDS[i]
        md.append(scale_mid(DEFAULT_MIDS_DEC[i], out_t, dec[i], min_keep))
    return enc, dec, me, md, sw


def transfer_rsu(src: nn.Module, dst: nn.Module,
                 keep_in: torch.Tensor, keep_out: torch.Tensor):
    src_out = src.rebnconvin.conv_s1.out_channels
    dst_out = dst.rebnconvin.conv_s1.out_channels
    src_mid = src.rebnconv1.conv_s1.out_channels
    dst_mid = dst.rebnconv1.conv_s1.out_channels
    mid_keep = shrink_idx(src_mid, dst_mid)
    assert keep_out.numel() == dst_out

    transfer_rebn(src.rebnconvin, dst.rebnconvin, keep_in, keep_out)
    transfer_rebn(src.rebnconv1, dst.rebnconv1, keep_out, mid_keep)

    for name, src_m in src.named_children():
        if name in ("rebnconvin", "rebnconv1") or name.startswith("pool"):
            continue
        if not isinstance(src_m, REBNCONV):
            continue
        dst_m = getattr(dst, name)
        sin = src_m.conv_s1.in_channels
        sout = src_m.conv_s1.out_channels
        din = dst_m.conv_s1.in_channels
        dout = dst_m.conv_s1.out_channels

        if sin == src_mid and sout == src_mid:
            ki, ko = mid_keep, mid_keep
        elif sin == 2 * src_mid and sout == src_mid:
            ki = torch.cat([mid_keep, mid_keep + src_mid], dim=0)
            ko = mid_keep
        elif sin == 2 * src_mid and sout == src_out:
            ki = torch.cat([mid_keep, mid_keep + src_mid], dim=0)
            ko = keep_out
        elif sin == src_out and sout == src_mid:
            ki, ko = keep_out, mid_keep
        else:
            ki = shrink_idx(sin, din)
            ko = keep_out if dout == keep_out.numel() else shrink_idx(sout, dout)

        if ki.numel() != din:
            ki = shrink_idx(sin, din)
        if ko.numel() != dout:
            ko = shrink_idx(sout, dout)
        transfer_rebn(src_m, dst_m, ki, ko)


def transfer_fuse(src: AsymBiChaFuseReduce, dst: AsymBiChaFuseReduce,
                  keep_high: torch.Tensor, keep_low: torch.Tensor):
    """keep_low also indexes fuse out channels (AsymBi: low_ch == out_ch)."""
    _slice_conv(src.feature_high[0], dst.feature_high[0], keep_low, keep_high)
    _slice_bn(src.feature_high[1], dst.feature_high[1], keep_low)

    sb = src.bottleneck_channels
    db = dst.bottleneck_channels
    b_keep = shrink_idx(sb, db)

    # topdown: [AvgPool, Conv out->btn, BN, ReLU, Conv btn->out, BN, Sigmoid]
    _slice_conv(src.topdown[1], dst.topdown[1], b_keep, keep_low)
    _slice_bn(src.topdown[2], dst.topdown[2], b_keep)
    _slice_conv(src.topdown[4], dst.topdown[4], keep_low, b_keep)
    _slice_bn(src.topdown[5], dst.topdown[5], keep_low)

    # bottomup: [Conv low->btn, BN, ReLU, SpatialAttention, Sigmoid]
    _slice_conv(src.bottomup[0], dst.bottomup[0], b_keep, keep_low)
    _slice_bn(src.bottomup[1], dst.bottomup[1], b_keep)
    with torch.no_grad():
        dst.bottomup[3].conv1.weight.data.copy_(src.bottomup[3].conv1.weight.data)

    _slice_conv(src.post[0], dst.post[0], keep_low, keep_low)
    _slice_bn(src.post[1], dst.post[1], keep_low)


def build_pruned_uiunet(
    teacher: UIUNET,
    keep_idx: Dict[int, torch.Tensor],
    min_keep: int = 4,
    in_ch: int = 3,
    out_ch: int = 1,
) -> UIUNET:
    enc, dec, me, md, sw = student_channel_config(keep_idx, min_keep=min_keep)
    print(f"[prune] teacher stage widths={list(TEACHER_STAGE_WIDTHS)}")
    print(f"[prune] student stage widths={sw}")
    print(f"[prune] student enc_outs={enc} dec_outs={dec}")
    print(f"[prune] student mids_enc={me} mids_dec={md}")

    student = UIUNET(
        in_ch=in_ch,
        out_ch=out_ch,
        enc_outs=enc,
        dec_outs=dec,
        mids_enc=me,
        mids_dec=md,
    )

    # encoder stages
    enc_names = ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
    for i, name in enumerate(enc_names):
        sid = ENC_STAGE_IDS[i]
        keep_out = keep_idx[sid]
        if i == 0:
            keep_in = torch.arange(in_ch)
        else:
            prev_sid = ENC_STAGE_IDS[i - 1]
            keep_in = keep_idx[prev_sid]
        transfer_rsu(getattr(teacher, name), getattr(student, name), keep_in, keep_out)

    # decoder stages
    dec_names = ["stage5d", "stage4d", "stage3d", "stage2d", "stage1d"]
    # inputs:
    # 5d: cat two e5  -> keep = cat(keep3, keep3+512)
    # 4d: cat two e4
    # 3d: cat two e3
    # 2d: cat two e2
    # 1d: cat(d2, e1) -> keep_idx[0] for both halves but different offsets
    t_e = list(DEFAULT_ENC_OUTS)
    t_d = list(DEFAULT_DEC_OUTS)
    for i, name in enumerate(dec_names):
        sid = DEC_STAGE_IDS[i]
        keep_out = keep_idx[sid]
        src = getattr(teacher, name)
        dst = getattr(student, name)
        sin = src.rebnconvin.conv_s1.in_channels
        if name == "stage1d":
            # teacher in = d2 + e1 = 64+64, both stage0
            k0 = keep_idx[0]
            keep_in = torch.cat([k0, k0 + t_d[3]], dim=0)  # d2 width then e1
        elif name == "stage5d":
            k = keep_idx[3]
            keep_in = torch.cat([k, k + t_e[4]], dim=0)
        elif name == "stage4d":
            k = keep_idx[3]
            keep_in = torch.cat([k, k + t_e[3]], dim=0)
        elif name == "stage3d":
            k = keep_idx[2]
            keep_in = torch.cat([k, k + t_e[2]], dim=0)
        elif name == "stage2d":
            k = keep_idx[1]
            keep_in = torch.cat([k, k + t_e[1]], dim=0)
        else:
            keep_in = shrink_idx(sin, dst.rebnconvin.conv_s1.in_channels)
        transfer_rsu(src, dst, keep_in, keep_out)

    # fuses
    transfer_fuse(teacher.fuse5, student.fuse5, keep_idx[3], keep_idx[3])  # high e6, low e5
    transfer_fuse(teacher.fuse4, student.fuse4, keep_idx[3], keep_idx[3])  # high d5, low e4
    transfer_fuse(teacher.fuse3, student.fuse3, keep_idx[2], keep_idx[2])  # high d4, low e3
    transfer_fuse(teacher.fuse2, student.fuse2, keep_idx[1], keep_idx[1])  # high d3, low e2

    # side heads
    side_keeps = [
        ("side1", keep_idx[0]),
        ("side2", keep_idx[0]),
        ("side3", keep_idx[1]),
        ("side4", keep_idx[2]),
        ("side5", keep_idx[3]),
        ("side6", keep_idx[3]),
    ]
    for name, kin in side_keeps:
        _slice_conv(getattr(teacher, name), getattr(student, name), None, kin)

    # outconv: 6 -> 1, copy as-is
    with torch.no_grad():
        student.outconv.weight.data.copy_(teacher.outconv.weight.data)
        if teacher.outconv.bias is not None:
            student.outconv.bias.data.copy_(teacher.outconv.bias.data)

    return student
