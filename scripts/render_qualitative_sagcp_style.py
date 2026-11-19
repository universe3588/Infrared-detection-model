#!/usr/bin/env python3
"""
按 SAGCP/常见 IRSTD 定性图风格重绘：
  - 顶部蓝色表头条
  - 绿=正确 / 黄=漏检 / 红=虚警
  - 角部局部放大
  - 紧凑通栏，便于 figure*[t] 留在正文

读取 qualitative_selection.csv，只推理选中样本（快）。
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DNANET = os.path.join(WORKSPACE, "DNANet")
DATA_ROOT = os.path.join(WORKSPACE, "RDANet-main", "datasets")
SEL_CSV = os.path.join(
    WORKSPACE,
    "面向边缘部署的红外弱小目标检测可迁移压缩框架",
    "images",
    "qualitative_selection.csv",
)
OUT_DIR = os.path.join(
    WORKSPACE,
    "面向边缘部署的红外弱小目标检测可迁移压缩框架",
    "images",
)

CKPTS = {
    "NUAA-SIRST": (
        os.path.join(
            DNANET,
            "result",
            "NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS",
            "mIoU__DNANet_NUAA-SIRST_epoch.pth.tar",
        ),
        os.path.join(
            DNANET,
            "compress_runs",
            "NUAA-SIRST_pipeline_20261029_181406",
            "student_best.pth.tar",
        ),
    ),
    "SIRST-V2": (
        os.path.join(
            DNANET,
            "result",
            "SIRST-V2_DNANet_14_10_2026_23_22_42_wDS",
            "mIoU__DNANet_SIRST-V2_epoch.pth.tar",
        ),
        os.path.join(
            DNANET,
            "compress_runs",
            "SIRST-V2_pipeline_20261029_200429",
            "student_best.pth.tar",
        ),
    ),
}

# SAGCP-like colors
C_HIT = (0, 200, 0)
C_MISS = (255, 220, 0)
C_FA = (255, 40, 40)
C_HEADER = (40, 90, 170)
BASE = 256
THR = 0.5
CELL = 168          # display cell size (compact)
HEADER_H = 28
GAP = 3
LEFT_LABEL_W = 78


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state(ckpt_path):
    ckpt = torch_load(ckpt_path)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "net", "student", "teacher"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt, ckpt[k]
        sample = next(iter(ckpt.values()))
        if torch.is_tensor(sample):
            return ckpt, ckpt
    raise ValueError(ckpt_path)


def connected_centroids(mask: np.ndarray):
    from scipy import ndimage

    lab, n = ndimage.label(mask.astype(np.uint8))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) == 0:
            continue
        out.append((float(ys.mean()), float(xs.mean()), int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))
    return out


def match_boxes(pred, gt, thr=3.0):
    pred_c = connected_centroids(pred)
    gt_c = connected_centroids(gt)
    used = set()
    hits, misses, fas = [], [], []
    for g in gt_c:
        best, best_d = None, 1e9
        for j, p in enumerate(pred_c):
            if j in used:
                continue
            d = ((g[0] - p[0]) ** 2 + (g[1] - p[1]) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, j
        if best is not None and best_d <= thr:
            used.add(best)
            hits.append(pred_c[best])
        else:
            misses.append(g)
    for j, p in enumerate(pred_c):
        if j not in used:
            fas.append(p)
    return hits, misses, fas


def to_gray_u8(img_rgb):
    g = img_rgb.mean(axis=2).astype(np.float32)
    g = g - g.min()
    g = g / (g.max() + 1e-6)
    return (g * 255).astype(np.uint8)


def add_zoom(panel, boxes, zoom=3, box_size=40, corner="tl"):
    h, w, _ = panel.shape
    if boxes:
        cy, cx = int(boxes[0][0]), int(boxes[0][1])
    else:
        cy, cx = h // 2, w // 2
    half = box_size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    crop = panel[y0:y1, x0:x1]
    if crop.size == 0:
        return panel
    z = np.array(Image.fromarray(crop).resize((max(1, crop.shape[1] * zoom), max(1, crop.shape[0] * zoom)), Image.NEAREST))
    out = panel.copy()
    zh, zw = z.shape[:2]
    m = 3
    if corner == "tl":
        yy0, xx0 = m, m
    else:
        yy0, xx0 = h - zh - m, w - zw - m
    out[yy0:yy0 + zh, xx0:xx0 + zw] = z
    # yellow-ish border like many IRSTD papers
    out[yy0 - 1:yy0 + zh + 1, xx0 - 1:xx0 + 1] = (255, 200, 0)
    out[yy0 - 1:yy0 + zh + 1, xx0 + zw - 1:xx0 + zw + 1] = (255, 200, 0)
    out[yy0 - 1:yy0 + 1, xx0 - 1:xx0 + zw + 1] = (255, 200, 0)
    out[yy0 + zh - 1:yy0 + zh + 1, xx0 - 1:xx0 + zw + 1] = (255, 200, 0)
    return out


def draw_boxes(rgb, hits, misses, fas):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    def _rect(b, color):
        y0, x0, y1, x1 = b[2], b[3], b[4], b[5]
        pad = 2
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=color, width=2)

    for b in hits:
        _rect(b, C_HIT)
    for b in misses:
        _rect(b, C_MISS)
    for b in fas:
        _rect(b, C_FA)
    return np.array(img)


def panel_ir(raw, focus_boxes):
    g = to_gray_u8(raw)
    rgb = np.stack([g] * 3, axis=-1)
    rgb = draw_boxes(rgb, focus_boxes, [], [])
    # use green for location on IR as "target region" reference — actually SAGCP puts boxes on all
    rgb = add_zoom(rgb, focus_boxes, corner="tl")
    return rgb


def panel_mask(mask_u8, hits, misses, fas, focus):
    rgb = np.stack([mask_u8] * 3, axis=-1)
    rgb = draw_boxes(rgb, hits, misses, fas)
    rgb = add_zoom(rgb, focus or hits or misses or fas, corner="tl")
    return rgb


def resize_cell(arr):
    return np.array(Image.fromarray(arr).resize((CELL, CELL), Image.BILINEAR))


def get_font(size=14):
    for fp in (
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


@torch.no_grad()
def predict(model, tensor, device):
    sys.path.insert(0, DNANET)
    from compress.losses import take_final

    x = tensor.unsqueeze(0).to(device)
    pred = torch.sigmoid(take_final(model(x)))[0, 0].cpu().numpy()
    return (pred >= THR).astype(np.uint8)


def build_models(dataset, device):
    sys.path.insert(0, DNANET)
    os.chdir(DNANET)
    from model.load_param_data import load_param
    from model.model_DNANet import DNANet, Res_CBAM_block

    tck, sck = CKPTS[dataset]
    nb, blocks = load_param("three", "resnet_18")
    teacher = DNANet(1, 3, Res_CBAM_block, list(blocks), list(nb), True).to(device).eval()
    _, st = load_state(tck)
    st = {k.replace("module.", ""): v for k, v in st.items()}
    teacher.load_state_dict(st, strict=False)

    ckpt, ss = load_state(sck)
    snb = list(ckpt["nb_filter"])
    student = DNANet(1, 3, Res_CBAM_block, list(blocks), snb, True).to(device).eval()
    ss = {k.replace("module.", ""): v for k, v in ss.items()}
    student.load_state_dict(ss, strict=False)
    for p in list(teacher.parameters()) + list(student.parameters()):
        p.requires_grad_(False)
    return teacher, student


def load_one(dataset, img_id, device):
    from torchvision import transforms
    from PIL import Image as PILImage

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_path = os.path.join(DATA_ROOT, dataset, "images", img_id + ".png")
    mask_path = os.path.join(DATA_ROOT, dataset, "masks", img_id + ".png")
    raw = np.array(PILImage.open(img_path).convert("RGB").resize((BASE, BASE), PILImage.BILINEAR))
    gt = np.array(PILImage.open(mask_path).convert("L").resize((BASE, BASE), PILImage.NEAREST))
    gt = (gt > 127).astype(np.uint8)
    tensor = tf(PILImage.fromarray(raw))
    return raw, gt, tensor


def compose_figure(rows: List[dict], out_path: str, col_titles=None):
    if col_titles is None:
        col_titles = ["Input", "GT", "Teacher", "Student"]
    n_rows = len(rows)
    n_cols = 4
    W = LEFT_LABEL_W + n_cols * CELL + (n_cols + 1) * GAP
    H = HEADER_H + n_rows * CELL + (n_rows + 1) * GAP + 22  # legend strip
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_h = get_font(15)
    font_l = get_font(11)
    font_s = get_font(10)

    # headers
    for j, title in enumerate(col_titles):
        x0 = LEFT_LABEL_W + GAP + j * (CELL + GAP)
        y0 = GAP
        draw.rectangle([x0, y0, x0 + CELL, y0 + HEADER_H], fill=C_HEADER)
        # center text
        bbox = draw.textbbox((0, 0), title, font=font_h)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (CELL - tw) / 2, y0 + (HEADER_H - th) / 2 - 1), title, fill=(255, 255, 255), font=font_h)

    y_base = GAP + HEADER_H + GAP
    for i, r in enumerate(rows):
        y0 = y_base + i * (CELL + GAP)
        # left label
        label = f"{r['dataset']}\n{r['id']}"
        # vertical-ish multiline centered in left gutter
        lines = label.split("\n")
        total_h = sum(draw.textbbox((0, 0), ln, font=font_l)[3] for ln in lines) + 2 * (len(lines) - 1)
        ty = y0 + (CELL - total_h) / 2
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font_l)
            tw = bbox[2] - bbox[0]
            draw.text(((LEFT_LABEL_W - tw) / 2, ty), ln, fill=(30, 30, 30), font=font_l)
            ty += bbox[3] - bbox[1] + 2

        panels = [r["ir"], r["gt"], r["tea"], r["stu"]]
        for j, p in enumerate(panels):
            x0 = LEFT_LABEL_W + GAP + j * (CELL + GAP)
            cell = Image.fromarray(resize_cell(p))
            canvas.paste(cell, (x0, y0))
            draw.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], outline=(180, 180, 180), width=1)

    # legend
    legend = "Green: correct   Yellow: miss   Red: false alarm   Corner: zoom-in"
    bbox = draw.textbbox((0, 0), legend, font=font_s)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, H - 18), legend, fill=(80, 80, 80), font=font_s)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))
    # also pdf via matplotlib for vector-ish embed
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.array(canvas))
    ax.axis("off")
    pdf = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[save] {out_path}")
    print(f"[save] {pdf}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] {device}")

    rows_sel = []
    with open(SEL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_sel.append(row)

    # Prefer: 2 NUAA success + 2 SIRST success + 2 fail  → one compact fig
    # And a second smaller fig optional — user wanted both in body; merge into ONE
    success = [r for r in rows_sel if r["role"] == "success"]
    fail = [r for r in rows_sel if r["role"] == "fail"]
    # take 2+2 success for compactness
    nuaa_s = [r for r in success if r["dataset"] == "NUAA-SIRST"][:2]
    v2_s = [r for r in success if r["dataset"] == "SIRST-V2"][:2]
    chosen = nuaa_s + v2_s + fail  # 6 rows but compact cells

    cache_models: Dict[str, Tuple] = {}
    rendered = []
    for r in chosen:
        ds, iid = r["dataset"], r["id"]
        if ds not in cache_models:
            cache_models[ds] = build_models(ds, device)
            print(f"[model] loaded {ds}")
        teacher, student = cache_models[ds]
        raw, gt, tensor = load_one(ds, iid, device)
        t_bin = predict(teacher, tensor, device)
        s_bin = predict(student, tensor, device)
        gt_boxes = connected_centroids(gt)
        t_hit, t_miss, t_fa = match_boxes(t_bin, gt)
        s_hit, s_miss, s_fa = match_boxes(s_bin, gt)
        focus = s_hit or t_hit or gt_boxes
        rendered.append({
            "dataset": ds,
            "id": iid,
            "ir": panel_ir(raw, focus),
            "gt": panel_mask((gt * 255).astype(np.uint8), gt_boxes, [], [], gt_boxes),
            "tea": panel_mask((t_bin * 255).astype(np.uint8), t_hit, t_miss, t_fa, focus),
            "stu": panel_mask((s_bin * 255).astype(np.uint8), s_hit, s_miss, s_fa, focus),
        })
        print(f"[ok] {ds}/{iid}")

    # single main qualitative figure (success+fail)
    compose_figure(rendered, os.path.join(OUT_DIR, "fig_qualitative.png"))

    # keep a slim fail-only figure as well (2 rows) for optional separate cite
    fail_rows = [x for x in rendered if any(f["id"] == x["id"] for f in fail)]
    if fail_rows:
        compose_figure(fail_rows, os.path.join(OUT_DIR, "fig_qualitative_fail.png"))


if __name__ == "__main__":
    main()
