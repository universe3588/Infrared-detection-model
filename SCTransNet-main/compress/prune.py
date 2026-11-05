from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from compress.adapters.sctrans_adapter import TEACHER_CHANNELS
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet, Res_block, CBN, UpBlock_attention


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
    return [int(keep_idx[s].numel()) for s in range(4)]


def _slice_conv(src: nn.Conv2d, dst: nn.Conv2d,
                keep_out: Optional[torch.Tensor], keep_in: Optional[torch.Tensor]):
    with torch.no_grad():
        w = src.weight.data
        if keep_out is not None:
            w = w[keep_out]
        if keep_in is not None:
            # depthwise: in_channels == groups, keep_in indexes groups
            if src.groups > 1 and src.groups == src.in_channels:
                # weight [out, 1, k, k] — only slice out
                pass
            else:
                w = w[:, keep_in]
        if w.shape != dst.weight.shape:
            # depthwise student/teacher groups differ: copy overlapping out channels
            if src.groups > 1 and dst.groups > 1 and keep_out is not None:
                n = min(w.shape[0], dst.weight.shape[0])
                dst.weight.data[:n].copy_(w[:n])
                return
            raise RuntimeError(
                f"conv slice mismatch {tuple(w.shape)} vs {tuple(dst.weight.shape)}"
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


def transfer_res_block(src: Res_block, dst: Res_block,
                       keep_in: Optional[torch.Tensor], keep_out: torch.Tensor):
    _slice_conv(src.conv1, dst.conv1, keep_out, keep_in)
    _slice_bn(src.bn1, dst.bn1, keep_out)
    _slice_conv(src.conv2, dst.conv2, keep_out, keep_out)
    _slice_bn(src.bn2, dst.bn2, keep_out)
    if src.shortcut is not None and dst.shortcut is not None:
        _slice_conv(src.shortcut[0], dst.shortcut[0], keep_out, keep_in)
        _slice_bn(src.shortcut[1], dst.shortcut[1], keep_out)


def transfer_sequential_res(src_seq: nn.Sequential, dst_seq: nn.Sequential,
                            keep_in: Optional[torch.Tensor], keep_out: torch.Tensor):
    assert len(src_seq) == len(dst_seq)
    for i, (sb, db) in enumerate(zip(src_seq, dst_seq)):
        if i == 0:
            transfer_res_block(sb, db, keep_in, keep_out)
        else:
            transfer_res_block(sb, db, keep_out, keep_out)


def transfer_cbn(src: CBN, dst: CBN, keep_in: torch.Tensor, keep_out: torch.Tensor):
    _slice_conv(src.conv, dst.conv, keep_out, keep_in)
    _slice_bn(src.norm, dst.norm, keep_out)


def transfer_up(src: UpBlock_attention, dst: UpBlock_attention,
                keep_low: torch.Tensor, keep_high: torch.Tensor, keep_out: torch.Tensor):
    """skip=keep_low, up=keep_high（同阶段时二者相同）, cat → nConvs → keep_out。"""
    with torch.no_grad():
        try:
            sx, dx = src.coatt.mlp_x[1], dst.coatt.mlp_x[1]
            w = sx.weight.data[keep_low][:, keep_low]
            if w.shape == dx.weight.shape:
                dx.weight.data.copy_(w)
                if sx.bias is not None and dx.bias is not None:
                    dx.bias.data.copy_(sx.bias.data[keep_low])
            sg, dg = src.coatt.mlp_g[1], dst.coatt.mlp_g[1]
            w2 = sg.weight.data[keep_low][:, keep_high]
            if w2.shape == dg.weight.shape:
                dg.weight.data.copy_(w2)
                if sg.bias is not None and dg.bias is not None:
                    dg.bias.data.copy_(sg.bias.data[keep_low])
        except Exception as e:
            print(f"[warn] CCA transfer fallback: {e}")

    t_low = int(src.coatt.mlp_x[1].in_features)
    keep_in_cat = torch.cat([keep_low, keep_high + t_low], dim=0)
    for i, (sc, dc) in enumerate(zip(src.nConvs, dst.nConvs)):
        if i == 0:
            transfer_cbn(sc, dc, keep_in_cat, keep_out)
        else:
            transfer_cbn(sc, dc, keep_out, keep_out)


def build_pruned_sctrans(
    teacher: SCTransNet,
    keep_idx: Dict[int, torch.Tensor],
    img_size: int = 256,
    mode: str = "train",
) -> SCTransNet:
    teacher_ch = list(getattr(teacher, "channel_num", TEACHER_CHANNELS))
    student_ch = student_channel_config(keep_idx)
    print(f"[prune] teacher channel_num={teacher_ch}")
    print(f"[prune] student channel_num={student_ch}")

    cfg = get_SCTrans_config()
    student = SCTransNet(
        cfg,
        n_channels=1,
        n_classes=1,
        img_size=img_size,
        mode=mode,
        deepsuper=True,
        channel_num=student_ch,
    )

    # CNN stem / encoder
    transfer_sequential_res(teacher.inc, student.inc, None, keep_idx[0])
    transfer_sequential_res(teacher.down_encoder1, student.down_encoder1, keep_idx[0], keep_idx[1])
    transfer_sequential_res(teacher.down_encoder2, student.down_encoder2, keep_idx[1], keep_idx[2])
    transfer_sequential_res(teacher.down_encoder3, student.down_encoder3, keep_idx[2], keep_idx[3])
    transfer_sequential_res(teacher.down_encoder4, student.down_encoder4, keep_idx[3], keep_idx[3])

    # Decoder ups
    transfer_up(teacher.up_decoder4, student.up_decoder4, keep_idx[3], keep_idx[3], keep_idx[2])
    transfer_up(teacher.up_decoder3, student.up_decoder3, keep_idx[2], keep_idx[2], keep_idx[1])
    transfer_up(teacher.up_decoder2, student.up_decoder2, keep_idx[1], keep_idx[1], keep_idx[0])
    transfer_up(teacher.up_decoder1, student.up_decoder1, keep_idx[0], keep_idx[0], keep_idx[0])

    _slice_conv(teacher.outc, student.outc, None, keep_idx[0])
    if teacher.deepsuper and student.deepsuper:
        _slice_conv(teacher.gt_conv5[0], student.gt_conv5[0], None, keep_idx[3])
        _slice_conv(teacher.gt_conv4[0], student.gt_conv4[0], None, keep_idx[2])
        _slice_conv(teacher.gt_conv3[0], student.gt_conv3[0], None, keep_idx[1])
        _slice_conv(teacher.gt_conv2[0], student.gt_conv2[0], None, keep_idx[0])
        with torch.no_grad():
            student.outconv.weight.data.copy_(teacher.outconv.weight.data)
            if teacher.outconv.bias is not None and student.outconv.bias is not None:
                student.outconv.bias.data.copy_(teacher.outconv.bias.data)

    # Transformer: 尽量按通道切片拷贝；失败则保留学生初始化（蒸馏恢复）
    try:
        _transfer_mtc(teacher.mtc, student.mtc, teacher_ch, keep_idx)
        print("[prune] mtc weights transferred (best-effort)")
    except Exception as e:
        print(f"[warn] mtc transfer skipped ({e}); student mtc keeps init")

    return student


def _transfer_mtc(src_mtc, dst_mtc, teacher_ch, keep_idx):
    """Best-effort: copy matching Conv/BN/LN slices for each scale."""
    # embeddings + reconstruct per scale
    for i, emb_s, emb_d, rec_s, rec_d in [
        (0, src_mtc.embeddings_1, dst_mtc.embeddings_1, src_mtc.reconstruct_1, dst_mtc.reconstruct_1),
        (1, src_mtc.embeddings_2, dst_mtc.embeddings_2, src_mtc.reconstruct_2, dst_mtc.reconstruct_2),
        (2, src_mtc.embeddings_3, dst_mtc.embeddings_3, src_mtc.reconstruct_3, dst_mtc.reconstruct_3),
        (3, src_mtc.embeddings_4, dst_mtc.embeddings_4, src_mtc.reconstruct_4, dst_mtc.reconstruct_4),
    ]:
        k = keep_idx[i]
        _slice_conv(emb_s.patch_embeddings, emb_d.patch_embeddings, k, k)
        _slice_conv(rec_s.conv, rec_d.conv, k, k)
        _slice_bn(rec_s.norm, rec_d.norm, k)

    # Concat keep for KV = cat(k0,k1,k2,k3) over teacher widths
    pieces = []
    cursor = 0
    for s, w in enumerate(teacher_ch):
        pieces.append(keep_idx[s] + cursor)
        cursor += w
    keep_kv = torch.cat(pieces, dim=0)

    # Walk encoder blocks: Attention + FFN channel dims
    for blk_s, blk_d in zip(src_mtc.encoder.layer, dst_mtc.encoder.layer):
        attn_s, attn_d = blk_s.channel_attn, blk_d.channel_attn
        for name, k in [("mhead1", keep_idx[0]), ("mhead2", keep_idx[1]),
                        ("mhead3", keep_idx[2]), ("mhead4", keep_idx[3]),
                        ("project_out1", keep_idx[0]), ("project_out2", keep_idx[1]),
                        ("project_out3", keep_idx[2]), ("project_out4", keep_idx[3])]:
            _slice_conv(getattr(attn_s, name), getattr(attn_d, name), k, k)
        _slice_conv(attn_s.mheadk, attn_d.mheadk, keep_kv, keep_kv)
        _slice_conv(attn_s.mheadv, attn_d.mheadv, keep_kv, keep_kv)
        # q/k/v depthwise: slice out channels only
        for name, k in [("q1", keep_idx[0]), ("q2", keep_idx[1]), ("q3", keep_idx[2]), ("q4", keep_idx[3])]:
            try:
                _slice_conv(getattr(attn_s, name), getattr(attn_d, name), k, None)
            except Exception:
                pass
        try:
            _slice_conv(attn_s.k, attn_d.k, keep_kv, None)
            _slice_conv(attn_s.v, attn_d.v, keep_kv, None)
        except Exception:
            pass

        for ffn_s, ffn_d, k in [
            (blk_s.ffn1, blk_d.ffn1, keep_idx[0]),
            (blk_s.ffn2, blk_d.ffn2, keep_idx[1]),
            (blk_s.ffn3, blk_d.ffn3, keep_idx[2]),
            (blk_s.ffn4, blk_d.ffn4, keep_idx[3]),
        ]:
            # project_in/out: dims change with expansion — only copy if shapes allow after slice
            try:
                # approximate: slice input/output channels; mid hidden may differ
                if ffn_s.project_out.out_channels == teacher_ch[0] or True:
                    # slice project_out out dim
                    po_s, po_d = ffn_s.project_out, ffn_d.project_out
                    # weight [out, in,1,1]
                    mid_d = po_d.in_channels
                    mid_s = po_s.in_channels
                    mid_keep = torch.arange(min(mid_s, mid_d))
                    w = po_s.weight.data[k][:, mid_keep]
                    if w.shape == po_d.weight.shape:
                        po_d.weight.data.copy_(w)
            except Exception:
                pass
