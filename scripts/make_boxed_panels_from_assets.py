# -*- coding: utf-8 -*-
"""用 定性对照/<样本>/ 里的原图生成拼图所需的带框面板（不从拼图裁切）。

输出（每个样本子文件夹）:
  Boxed_Input.png / Boxed_GT.png / Boxed_Teacher.png / Boxed_Student.png
颜色约定与正文一致: 绿=correct, 黄=miss, 红=false alarm；角部 zoom-in。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"d:\Desktop\红外检测模型\面向边缘部署的红外弱小目标检测可迁移压缩框架\images\定性对照")

FIG6 = [
    "NUAA-SIRST_Misc_423",
    "NUAA-SIRST_Misc_129",
    "SIRST-V2_S20210422_S7_1",
    "SIRST-V2_Misc_38",
    "NUAA-SIRST_Misc_229",
    "SIRST-V2_foo-00112",
]

C_HIT = (0, 200, 0)
C_MISS = (255, 220, 0)
C_FA = (255, 40, 40)
MATCH_THR = 3.0


def connected_centroids(mask: np.ndarray):
    from scipy import ndimage

    lab, n = ndimage.label(mask.astype(np.uint8))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) == 0:
            continue
        out.append(
            (
                float(ys.mean()),
                float(xs.mean()),
                int(ys.min()),
                int(xs.min()),
                int(ys.max()),
                int(xs.max()),
            )
        )
    return out


def match_boxes(pred, gt, thr=MATCH_THR):
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


def draw_boxes(rgb: np.ndarray, hits, misses, fas) -> np.ndarray:
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


def add_zoom(panel: np.ndarray, boxes, zoom=3, box_size=40, corner="tl") -> np.ndarray:
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
    z = np.array(
        Image.fromarray(crop).resize(
            (max(1, crop.shape[1] * zoom), max(1, crop.shape[0] * zoom)),
            Image.NEAREST,
        )
    )
    out = panel.copy()
    zh, zw = z.shape[:2]
    m = 3
    if corner == "tl":
        yy0, xx0 = m, m
    else:
        yy0, xx0 = h - zh - m, w - zw - m
    # keep zoom inside canvas
    yy0 = max(1, min(yy0, h - zh - 1))
    xx0 = max(1, min(xx0, w - zw - 1))
    out[yy0 : yy0 + zh, xx0 : xx0 + zw] = z
    out[yy0 - 1 : yy0 + zh + 1, xx0 - 1 : xx0 + 1] = (255, 200, 0)
    out[yy0 - 1 : yy0 + zh + 1, xx0 + zw - 1 : xx0 + zw + 1] = (255, 200, 0)
    out[yy0 - 1 : yy0 + 1, xx0 - 1 : xx0 + zw + 1] = (255, 200, 0)
    out[yy0 + zh - 1 : yy0 + zh + 1, xx0 - 1 : xx0 + zw + 1] = (255, 200, 0)
    return out


def load_gray_u8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im)


def to_rgb_gray(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    g = g - g.min()
    g = g / (g.max() + 1e-6)
    g8 = (g * 255).astype(np.uint8)
    return np.stack([g8] * 3, axis=-1)


def mask_to_rgb(mask01: np.ndarray) -> np.ndarray:
    u8 = (mask01.astype(np.uint8) * 255)
    return np.stack([u8] * 3, axis=-1)


def process_one(folder: Path) -> None:
    inp = load_gray_u8(folder / "Input.png")
    gt = (load_gray_u8(folder / "GT.png") > 127).astype(np.uint8)
    tea = (load_gray_u8(folder / "Teacher.png") > 127).astype(np.uint8)
    stu = (load_gray_u8(folder / "Student.png") > 127).astype(np.uint8)

    gt_boxes = connected_centroids(gt)
    t_hit, t_miss, t_fa = match_boxes(tea, gt)
    s_hit, s_miss, s_fa = match_boxes(stu, gt)
    focus = s_hit or t_hit or gt_boxes

    # Input: 目标位置用绿框（相对 GT 焦点），+ zoom
    boxed_input = add_zoom(draw_boxes(to_rgb_gray(inp), focus, [], []), focus)
    # GT: 绿框标出 GT 连通域
    boxed_gt = add_zoom(draw_boxes(mask_to_rgb(gt), gt_boxes, [], []), gt_boxes)
    # Teacher / Student: hit/miss/fa
    boxed_tea = add_zoom(
        draw_boxes(mask_to_rgb(tea), t_hit, t_miss, t_fa),
        focus or t_hit or t_miss or t_fa or gt_boxes,
    )
    boxed_stu = add_zoom(
        draw_boxes(mask_to_rgb(stu), s_hit, s_miss, s_fa),
        focus or s_hit or s_miss or s_fa or gt_boxes,
    )

    Image.fromarray(boxed_input).save(folder / "Boxed_Input.png")
    Image.fromarray(boxed_gt).save(folder / "Boxed_GT.png")
    Image.fromarray(boxed_tea).save(folder / "Boxed_Teacher.png")
    Image.fromarray(boxed_stu).save(folder / "Boxed_Student.png")

    # 删除此前从拼图裁切的 Panel_*，避免混淆
    for old in folder.glob("Panel_*.png"):
        old.unlink()
        print(f"[del] {old.name}")

    print(
        f"[ok] {folder.name}  "
        f"T(hit/miss/fa)={len(t_hit)}/{len(t_miss)}/{len(t_fa)}  "
        f"S(hit/miss/fa)={len(s_hit)}/{len(s_miss)}/{len(s_fa)}"
    )


def write_readme() -> None:
    text = """正文 fig_qualitative 用到的素材（按图中从上到下）
路径: images/定性对照/<数据集>_<样本ID>/

每个子文件夹:
  【原图，自行排版用】
    Input.png / Input_RGB.png / GT.png / Teacher.png / Student.png
  【由原图生成的带框面板，拼图风格】
    Boxed_Input.png / Boxed_GT.png / Boxed_Teacher.png / Boxed_Student.png
    绿=correct  黄=miss  红=false alarm  角部=zoom-in

正文 6 张:
  1. NUAA-SIRST_Misc_423
  2. NUAA-SIRST_Misc_129
  3. SIRST-V2_S20210422_S7_1
  4. SIRST-V2_Misc_38
  5. NUAA-SIRST_Misc_229
  6. SIRST-V2_foo-00112

生成脚本: scripts/make_boxed_panels_from_assets.py
"""
    (ROOT / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    for name in FIG6:
        d = ROOT / name
        if not d.is_dir():
            raise SystemExit(f"missing folder: {d}")
        for need in ("Input.png", "GT.png", "Teacher.png", "Student.png"):
            if not (d / need).is_file():
                raise SystemExit(f"missing asset: {d / need}")
        process_one(d)
    write_readme()
    print("[done]", ROOT)


if __name__ == "__main__":
    main()
