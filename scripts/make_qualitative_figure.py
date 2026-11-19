#!/usr/bin/env python3
"""
生成论文定性图（Ex15/Ex16）：IR | GT | Teacher | Student
版式参考常见 IRSTD 可视化（局部放大 + 红/黄/蓝框），列仅为师生对照。

依赖：Python + torch + numpy + matplotlib + Pillow（无需额外商业软件）

示例：
  python scripts/make_qualitative_figure.py --device cpu
  python scripts/make_qualitative_figure.py --device cuda --datasets NUAA-SIRST,SIRST-V2
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dnanet-root", default="DNANet")
    p.add_argument("--data-root", default="RDANet-main/datasets")
    p.add_argument("--datasets", default="NUAA-SIRST,SIRST-V2")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--base-size", type=int, default=256)
    p.add_argument("--n-success", type=int, default=3, help="每集成功/典型例数量")
    p.add_argument("--n-fail", type=int, default=1, help="每集失败例数量（师生都差或漏检）")
    p.add_argument(
        "--out",
        default="面向边缘部署的红外弱小目标检测可迁移压缩框架/images/fig_qualitative.png",
    )
    p.add_argument("--out-fail", default="面向边缘部署的红外弱小目标检测可迁移压缩框架/images/fig_qualitative_fail.png")
    # default ckpts (workspace-relative)
    p.add_argument(
        "--teacher-nuaa",
        default="DNANet/result/NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar",
    )
    p.add_argument(
        "--student-nuaa",
        default="DNANet/compress_runs/NUAA-SIRST_pipeline_20261029_181406/student_best.pth.tar",
    )
    p.add_argument(
        "--teacher-sirstv2",
        default="DNANet/result/SIRST-V2_DNANet_14_10_2026_23_22_42_wDS/mIoU__DNANet_SIRST-V2_epoch.pth.tar",
    )
    p.add_argument(
        "--student-sirstv2",
        default="DNANet/compress_runs/SIRST-V2_pipeline_20261029_200429/student_best.pth.tar",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_path(workspace: str, p: str) -> str:
    cands = [p, os.path.join(workspace, p), os.path.abspath(p)]
    for c in cands:
        if os.path.isfile(c) or os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(workspace, p))


def load_state(ckpt_path: str) -> dict:
    ckpt = torch_load(ckpt_path)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "net", "student", "teacher"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt, ckpt[k]
        if all(torch.is_tensor(v) for v in list(ckpt.values())[:3]):
            return ckpt, ckpt
    raise ValueError(f"bad ckpt: {ckpt_path}")


def build_model(dnanet_root: str, nb_filter, num_blocks, device):
    from model.model_DNANet import DNANet, Res_CBAM_block

    model = DNANet(
        num_classes=1,
        input_channels=3,
        block=Res_CBAM_block,
        num_blocks=list(num_blocks),
        nb_filter=list(nb_filter),
        deep_supervision=True,
    )
    return model.to(device).eval()


def load_teacher(dnanet_root, ckpt_path, device):
    from model.load_param_data import load_param

    nb, blocks = load_param("three", "resnet_18")
    model = build_model(dnanet_root, nb, blocks, device)
    _, state = load_state(ckpt_path)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(dnanet_root, ckpt_path, device):
    from model.load_param_data import load_param

    _, blocks = load_param("three", "resnet_18")
    ckpt, state = load_state(ckpt_path)
    nb = ckpt.get("nb_filter")
    if not nb:
        raise ValueError(f"student ckpt missing nb_filter: {ckpt_path}")
    model = build_model(dnanet_root, nb, blocks, device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def take_final(preds):
    if isinstance(preds, (list, tuple)):
        return preds[-1]
    return preds


def connected_centroids(mask: np.ndarray) -> List[Tuple[float, float, int, int, int, int]]:
    """Return list of (cy, cx, y0, x0, y1, x1) for connected components >0."""
    from scipy import ndimage

    lab, n = ndimage.label(mask.astype(np.uint8))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) == 0:
            continue
        out.append((float(ys.mean()), float(xs.mean()), int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))
    return out


def match_pd_fa(pred: np.ndarray, gt: np.ndarray, thr: float = 3.0):
    """Centroid matching; return n_hit, n_gt, n_fa_comp, hit_boxes, miss_boxes, fa_boxes."""
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
    return hits, misses, fas, len(hits), len(gt_c)


def local_scr(img: np.ndarray, gt: np.ndarray) -> float:
    """Rough SCR: (mu_t - mu_b) / (sigma_b + eps) on grayscale."""
    if img.ndim == 3:
        g = img.mean(axis=2)
    else:
        g = img
    m = gt > 0
    if m.sum() < 1:
        return 0.0
    # dilate background ring
    from scipy import ndimage

    ring = ndimage.binary_dilation(m, iterations=5) & (~m)
    if ring.sum() < 5:
        ring = ~m
    mu_t = float(g[m].mean())
    mu_b = float(g[ring].mean())
    sig = float(g[ring].std()) + 1e-6
    return (mu_t - mu_b) / sig


def pixel_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union + 1e-6)


def to_gray_u8(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 3:
        g = img_rgb.mean(axis=2)
    else:
        g = img_rgb
    g = g.astype(np.float32)
    g = g - g.min()
    g = g / (g.max() + 1e-6)
    return (g * 255).astype(np.uint8)


def draw_panel(base_gray: np.ndarray, mask: Optional[np.ndarray], boxes_hit, boxes_miss, boxes_fa, title_mode: str):
    """Compose RGB panel: gray IR or binary mask + colored boxes."""
    h, w = base_gray.shape
    if title_mode in ("pred", "gt"):
        rgb = np.stack([base_gray] * 3, axis=-1)
        if mask is not None:
            # white target on black already if mask; keep as is
            pass
    else:
        rgb = np.stack([base_gray] * 3, axis=-1)

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    # colors: hit red, miss yellow, fa blue
    for b in boxes_hit:
        y0, x0, y1, x1 = b[2], b[3], b[4], b[5]
        pad = 2
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(255, 0, 0), width=2)
    for b in boxes_miss:
        y0, x0, y1, x1 = b[2], b[3], b[4], b[5]
        pad = 2
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(255, 220, 0), width=2)
    for b in boxes_fa:
        y0, x0, y1, x1 = b[2], b[3], b[4], b[5]
        pad = 2
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(0, 120, 255), width=2)
    return np.array(img)


def add_zoom(panel: np.ndarray, boxes, zoom: int = 3, box_size: int = 48) -> np.ndarray:
    """Put a zoom inset at bottom-right around first available box center."""
    h, w, _ = panel.shape
    if not boxes:
        # center crop
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(boxes[0][0]), int(boxes[0][1])
    half = box_size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    crop = panel[y0:y1, x0:x1]
    if crop.size == 0:
        return panel
    z = np.array(Image.fromarray(crop).resize((crop.shape[1] * zoom, crop.shape[0] * zoom), Image.NEAREST))
    out = panel.copy()
    zh, zw = z.shape[:2]
    # bottom-right with margin
    m = 4
    out[h - zh - m : h - m, w - zw - m : w - m] = z
    # border
    out[h - zh - m - 1 : h - m + 1, w - zw - m - 1 : w - zw - m + 1] = (255, 80, 80)
    out[h - zh - m - 1 : h - m + 1, w - m - 1 : w - m + 1] = (255, 80, 80)
    out[h - zh - m - 1 : h - zh - m + 1, w - zw - m - 1 : w - m + 1] = (255, 80, 80)
    out[h - m - 1 : h - m + 1, w - zw - m - 1 : w - m + 1] = (255, 80, 80)
    return out


@torch.no_grad()
def predict_mask(model, img_tensor, thr, device):
    from compress.losses import take_final as _tf

    x = img_tensor.unsqueeze(0).to(device)
    pred = torch.sigmoid(_tf(model(x)))[0, 0].cpu().numpy()
    return (pred >= thr).astype(np.uint8), pred


def eval_dataset(args, workspace, dataset: str, teacher_ckpt: str, student_ckpt: str, device):
    dnanet_root = resolve_path(workspace, args.dnanet_root)
    sys.path.insert(0, dnanet_root)
    os.chdir(dnanet_root)

    from torchvision import transforms
    from model.load_param_data import load_dataset
    from model.utils import TestSetLoader
    from torch.utils.data import DataLoader

    data_root = resolve_path(workspace, args.data_root)
    teacher = load_teacher(dnanet_root, resolve_path(workspace, teacher_ckpt), device)
    student = load_student(dnanet_root, resolve_path(workspace, student_ckpt), device)

    _, ids, _ = load_dataset(data_root, dataset, "50_50", 0, 42)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = TestSetLoader(
        os.path.join(data_root, dataset),
        img_id=ids,
        base_size=args.base_size,
        crop_size=args.base_size,
        transform=tf,
        suffix=".png",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    records = []
    for i, (data, labels) in enumerate(loader):
        img_id = ids[i]
        # raw image for display
        raw_path = os.path.join(data_root, dataset, "images", img_id + ".png")
        raw = np.array(Image.open(raw_path).convert("RGB").resize((args.base_size, args.base_size), Image.BILINEAR))
        gt = (labels[0, 0].numpy() > 0.5).astype(np.uint8)

        t_bin, _ = predict_mask(teacher, data[0], args.thr, device)
        s_bin, _ = predict_mask(student, data[0], args.thr, device)

        t_hit, t_miss, t_fa, t_nh, t_ng = match_pd_fa(t_bin, gt)
        s_hit, s_miss, s_fa, s_nh, s_ng = match_pd_fa(s_bin, gt)
        scr = local_scr(raw, gt)
        area = int(gt.sum())
        t_iou = pixel_iou(t_bin, gt)
        s_iou = pixel_iou(s_bin, gt)

        records.append({
            "dataset": dataset,
            "id": img_id,
            "raw": raw,
            "gt": gt,
            "t_bin": t_bin,
            "s_bin": s_bin,
            "t_hit": t_hit, "t_miss": t_miss, "t_fa": t_fa,
            "s_hit": s_hit, "s_miss": s_miss, "s_fa": s_fa,
            "scr": scr,
            "area": area,
            "t_iou": t_iou,
            "s_iou": s_iou,
            "t_pd": t_nh / max(t_ng, 1),
            "s_pd": s_nh / max(s_ng, 1),
        })
        if (i + 1) % 20 == 0:
            print(f"[{dataset}] {i+1}/{len(ids)}")

    return records


def pick_cases(records: List[dict], n_success: int, n_fail: int):
    # success: both Pd==1 and IoU decent; prefer lower SCR / smaller area (harder)
    succ = [r for r in records if r["t_pd"] >= 1.0 and r["s_pd"] >= 1.0 and r["t_iou"] > 0.3 and r["s_iou"] > 0.3]
    succ = sorted(succ, key=lambda r: (r["scr"], r["area"]))
    success = succ[:n_success]

    # fail: miss or very low IoU on student or teacher; prefer low SCR
    fail = [r for r in records if r["s_pd"] < 1.0 or r["t_pd"] < 1.0 or r["s_iou"] < 0.15]
    fail = sorted(fail, key=lambda r: (r["scr"], -len(r["s_miss"]) - len(r["t_miss"])))
    # avoid duplicates with success
    sid = {r["id"] for r in success}
    fail = [r for r in fail if r["id"] not in sid][:n_fail]

    # if not enough success, fill with best IoU
    if len(success) < n_success:
        rest = sorted(records, key=lambda r: -(r["t_iou"] + r["s_iou"]) / 2)
        for r in rest:
            if r["id"] not in sid:
                success.append(r)
                sid.add(r["id"])
            if len(success) >= n_success:
                break
    return success, fail


def render_row(r: dict):
    gray = to_gray_u8(r["raw"])
    gt_vis = (r["gt"] * 255).astype(np.uint8)
    t_vis = (r["t_bin"] * 255).astype(np.uint8)
    s_vis = (r["s_bin"] * 255).astype(np.uint8)

    # IR with GT boxes as reference hits on IR for location
    ir = draw_panel(gray, None, r["s_hit"] or r["t_hit"] or connected_centroids(r["gt"]), [], [], "ir")
    ir = add_zoom(ir, r["s_hit"] or r["t_hit"] or connected_centroids(r["gt"]))

    gt_p = draw_panel(gt_vis, r["gt"], connected_centroids(r["gt"]), [], [], "gt")
    gt_p = add_zoom(gt_p, connected_centroids(r["gt"]))

    tea = draw_panel(t_vis, r["t_bin"], r["t_hit"], r["t_miss"], r["t_fa"], "pred")
    tea = add_zoom(tea, r["t_hit"] or r["t_miss"] or r["t_fa"] or connected_centroids(r["gt"]))

    stu = draw_panel(s_vis, r["s_bin"], r["s_hit"], r["s_miss"], r["s_fa"], "pred")
    stu = add_zoom(stu, r["s_hit"] or r["s_miss"] or r["s_fa"] or connected_centroids(r["gt"]))
    return [ir, gt_p, tea, stu]


def save_grid(rows_panels: List[List[np.ndarray]], row_labels: List[str], out_path: str, col_titles=None):
    import matplotlib.pyplot as plt

    if col_titles is None:
        col_titles = ["IR image", "GT", "Teacher", "Student"]
    n_rows = len(rows_panels)
    n_cols = 4
    fig_w = 2.4 * n_cols
    fig_h = 2.4 * n_rows + 0.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(rows_panels[i][j])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11, pad=4)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=8, rotation=90, labelpad=6)
    # legend
    fig.text(
        0.5, 0.01,
        "Box: red=correct, yellow=miss, blue=false alarm; inset=local zoom",
        ha="center", fontsize=8, color="0.35",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, bbox_inches="tight", facecolor="w")
    pdf = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf, bbox_inches="tight", facecolor="w")
    plt.close()
    print(f"[save] {out_path}")
    print(f"[save] {pdf}")


def main():
    args = parse_args()
    workspace = os.getcwd()
    # if launched from elsewhere, prefer script-relative workspace
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(here, "DNANet")):
        workspace = here

    device = resolve_device(args.device)
    print(f"[env] device={device}  workspace={workspace}")

    # scipy needed for label/dilation
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise SystemExit("需要 scipy：pip install scipy")

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ckpt_map = {
        "NUAA-SIRST": (args.teacher_nuaa, args.student_nuaa),
        "SIRST-V2": (args.teacher_sirstv2, args.student_sirstv2),
    }

    all_success, all_fail = [], []
    for ds in ds_list:
        if ds not in ckpt_map:
            print(f"[skip] no default ckpt for {ds}")
            continue
        tck, sck = ckpt_map[ds]
        if not os.path.isfile(resolve_path(workspace, tck)) or not os.path.isfile(resolve_path(workspace, sck)):
            print(f"[skip] missing ckpt for {ds}")
            continue
        print(f"[run] {ds}")
        recs = eval_dataset(args, workspace, ds, tck, sck, device)
        succ, fail = pick_cases(recs, args.n_success, args.n_fail)
        all_success.extend(succ)
        all_fail.extend(fail)
        print(f"[{ds}] pick success={[r['id'] for r in succ]} fail={[r['id'] for r in fail]}")

    if not all_success and not all_fail:
        raise SystemExit("no cases selected")

    if all_success:
        panels = [render_row(r) for r in all_success]
        labels = [f"{r['dataset']}\n{r['id']}\nSCR={r['scr']:.2f}" for r in all_success]
        out = args.out if os.path.isabs(args.out) else os.path.join(workspace, args.out)
        save_grid(panels, labels, out)

    if all_fail:
        panels = [render_row(r) for r in all_fail]
        labels = [f"{r['dataset']}\n{r['id']}\nFAIL SCR={r['scr']:.2f}" for r in all_fail]
        outf = args.out_fail if os.path.isabs(args.out_fail) else os.path.join(workspace, args.out_fail)
        save_grid(panels, labels, outf)

    # also write a small selection csv
    sel = os.path.join(workspace, "面向边缘部署的红外弱小目标检测可迁移压缩框架/images/qualitative_selection.csv")
    os.makedirs(os.path.dirname(sel), exist_ok=True)
    with open(sel, "w", encoding="utf-8") as f:
        f.write("role,dataset,id,scr,area,t_iou,s_iou,t_pd,s_pd\n")
        for role, arr in (("success", all_success), ("fail", all_fail)):
            for r in arr:
                f.write(
                    f"{role},{r['dataset']},{r['id']},{r['scr']:.4f},{r['area']},"
                    f"{r['t_iou']:.4f},{r['s_iou']:.4f},{r['t_pd']:.4f},{r['s_pd']:.4f}\n"
                )
    print(f"[save] {sel}")


if __name__ == "__main__":
    main()
