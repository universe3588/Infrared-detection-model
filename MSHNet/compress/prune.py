from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from compress.adapters.mshnet_adapter import MSH_IN_LAYOUT, TEACHER_CHANNELS
from model.MSHNet import MSHNet, ResNet, DEFAULT_BLOCKS


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


def student_channel_config(keep_idx: Dict[int, torch.Tensor]) -> List[int]:
    return [int(keep_idx[s].numel()) for s in range(5)]


def _slice_conv(src: nn.Conv2d, dst: nn.Conv2d, keep_out: Optional[torch.Tensor], keep_in: Optional[torch.Tensor]):
    with torch.no_grad():
        w = src.weight.data
        if keep_out is not None:
            w = w[keep_out]
        if keep_in is not None:
            w = w[:, keep_in]
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


def _slice_ca(src, dst, keep_out: torch.Tensor):
    with torch.no_grad():
        try:
            src_mid = src.fc1.out_channels
            dst_mid = dst.fc1.out_channels
            mid_keep = torch.arange(min(src_mid, dst_mid))
            w1 = src.fc1.weight.data[mid_keep][:, keep_out]
            if w1.shape == dst.fc1.weight.shape:
                dst.fc1.weight.data.copy_(w1)
            w2 = src.fc2.weight.data[keep_out][:, mid_keep]
            if w2.shape == dst.fc2.weight.shape:
                dst.fc2.weight.data.copy_(w2)
        except Exception as e:
            print(f"[warn] CA slice fallback ({e}); student CA keeps init")


def transfer_res_block(src: nn.Module, dst: nn.Module, keep_in: Optional[torch.Tensor], keep_out: torch.Tensor):
    _slice_conv(src.conv1, dst.conv1, keep_out, keep_in)
    _slice_bn(src.bn1, dst.bn1, keep_out)
    _slice_conv(src.conv2, dst.conv2, keep_out, keep_out)
    _slice_bn(src.bn2, dst.bn2, keep_out)
    _slice_ca(src.ca, dst.ca, keep_out)
    with torch.no_grad():
        dst.sa.conv1.weight.data.copy_(src.sa.conv1.weight.data)
    if src.shortcut is not None and dst.shortcut is not None:
        _slice_conv(src.shortcut[0], dst.shortcut[0], keep_out, keep_in)
        _slice_bn(src.shortcut[1], dst.shortcut[1], keep_out)


def build_keep_in(
    layout: List[Optional[int]],
    teacher_ch: Sequence[int],
    keep_idx: Dict[int, torch.Tensor],
) -> torch.Tensor:
    pieces = []
    cursor = 0
    for sid in layout:
        if sid is None:
            pieces.append(torch.arange(3) + cursor)
            cursor += 3
            continue
        width = teacher_ch[sid]
        local = keep_idx[sid]
        pieces.append(local + cursor)
        cursor += width
    return torch.cat(pieces, dim=0)


def transfer_sequential_blocks(
    src_seq: nn.Sequential,
    dst_seq: nn.Sequential,
    keep_in: Optional[torch.Tensor],
    keep_out: torch.Tensor,
):
    assert len(src_seq) == len(dst_seq)
    for i, (sb, db) in enumerate(zip(src_seq, dst_seq)):
        if i == 0:
            transfer_res_block(sb, db, keep_in, keep_out)
        else:
            transfer_res_block(sb, db, keep_out, keep_out)


def build_pruned_mshnet(
    teacher: MSHNet,
    keep_idx: Dict[int, torch.Tensor],
    input_channels: int = 3,
) -> MSHNet:
    teacher_ch = list(getattr(teacher, "param_channels", TEACHER_CHANNELS))
    student_ch = student_channel_config(keep_idx)
    print(f"[prune] teacher channels={teacher_ch}")
    print(f"[prune] student channels={student_ch}")

    student = MSHNet(
        input_channels,
        block=ResNet,
        channels=student_ch,
        blocks=list(getattr(teacher, "param_blocks", DEFAULT_BLOCKS)),
    )

    # conv_init
    _slice_conv(teacher.conv_init, student.conv_init, keep_idx[0], None)

    layer_out_stage = {
        "encoder_0": 0,
        "encoder_1": 1,
        "encoder_2": 2,
        "encoder_3": 3,
        "middle_layer": 4,
        "decoder_3": 3,
        "decoder_2": 2,
        "decoder_1": 1,
        "decoder_0": 0,
    }
    for name, out_stage in layer_out_stage.items():
        src = getattr(teacher, name)
        dst = getattr(student, name)
        layout = MSH_IN_LAYOUT[name]
        keep_in = build_keep_in(layout, teacher_ch, keep_idx)
        keep_out = keep_idx[out_stage]
        transfer_sequential_blocks(src, dst, keep_in, keep_out)

    for i, oname in enumerate(["output_0", "output_1", "output_2", "output_3"]):
        _slice_conv(getattr(teacher, oname), getattr(student, oname), None, keep_idx[i])

    # final: 4->1 fused side masks, structure unchanged
    with torch.no_grad():
        student.final.weight.data.copy_(teacher.final.weight.data)
        if teacher.final.bias is not None and student.final.bias is not None:
            student.final.bias.data.copy_(teacher.final.bias.data)

    return student
