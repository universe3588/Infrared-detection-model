# -*- coding: utf-8 -*-
"""Fix qualitative legend typo; replot Pareto / criterion / distill bars."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\Desktop\红外检测模型\面向边缘部署的红外弱小目标检测可迁移压缩框架")


def fix_qualitative_legend() -> None:
    img_path = ROOT / "images" / "fig_qualitative.png"
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    band_top = h - 28
    draw.rectangle([0, band_top, w, h], fill=(255, 255, 255))
    text = "Green: correct   Yellow: miss   Red: false alarm   Corner: zoom-in"
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = band_top + (28 - th) // 2 - 1
    draw.text((x, y), text, fill=(20, 20, 20), font=font)
    im.save(img_path)
    print("fixed qualitative legend ->", img_path)


def copy_out(src: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        p = src.with_suffix(f".{ext}")
        if p.exists():
            shutil.copy2(p, ROOT / "images" / f"{name}.{ext}")


def plot_pareto() -> None:
    csv_path = (
        ROOT / "images" / "Matlab" / "精度-复杂度Pareto曲线" / "data" / "pareto_teacher_student.csv"
    )
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    order = ["DNA-Net", "UIU-Net", "MSHNet", "SCTransNet"]
    colors = {
        "DNA-Net": (0.75, 0.20, 0.20),
        "UIU-Net": (0.20, 0.45, 0.75),
        "MSHNet": (0.15, 0.55, 0.30),
        "SCTransNet": (0.55, 0.30, 0.65),
    }
    data: dict[str, dict[str, dict]] = {}
    for r in rows:
        data.setdefault(r["backbone"], {})[r["role"]] = r

    def get(bb: str, role: str, key: str) -> float:
        return float(data[bb][role][key])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=160)
    fig.suptitle(
        "Accuracy--Complexity Pareto (Teacher → Student)",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    for bb in order:
        c = colors[bb]
        xp = [get(bb, "teacher", "params_M"), get(bb, "student", "params_M")]
        yp = [get(bb, "teacher", "iou_mean"), get(bb, "student", "iou_mean")]
        ax.plot(xp, yp, "-", color=c, lw=1.8, label=bb)
        ax.plot(xp[0], yp[0], "o", ms=8, mfc=c, mec="k", mew=0.8)
        ax.plot(xp[1], yp[1], "s", ms=8, mfc="w", mec=c, mew=1.6)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 80)
    ax.set_ylim(0.70, 0.80)
    ax.set_xlabel("Params (M)")
    ax.set_ylabel("Mean IoU")
    ax.set_title("(a) IoU--Params")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax = axes[1]
    for bb in order:
        c = colors[bb]
        xp = [get(bb, "teacher", "flops_G"), get(bb, "student", "flops_G")]
        yp = [get(bb, "teacher", "iou_mean"), get(bb, "student", "iou_mean")]
        ax.plot(xp, yp, "-", color=c, lw=1.8, label=bb)
        ax.plot(xp[0], yp[0], "o", ms=8, mfc=c, mec="k", mew=0.8)
        ax.plot(xp[1], yp[1], "s", ms=8, mfc="w", mec=c, mew=1.6)
    ax.plot([], [], "o", mfc="0.4", mec="k", label="Teacher")
    ax.plot([], [], "s", mfc="w", mec="0.2", mew=1.6, label="Student")
    ax.set_xscale("log")
    ax.set_xlim(3, 150)
    ax.set_ylim(0.70, 0.80)
    ax.set_xlabel("FLOPs (G)")
    ax.set_ylabel("Mean IoU")
    ax.set_title("(b) IoU--FLOPs")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir = ROOT / "images" / "Matlab" / "精度-复杂度Pareto曲线" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig_pareto"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    copy_out(stem, "fig_pareto")
    plt.close(fig)
    print("pareto saved")


def plot_criterion() -> None:
    crit_path = ROOT / "images" / "Matlab" / "敲除校准曲线" / "data" / "criterion_nuaa.csv"
    crit = list(csv.DictReader(crit_path.open(encoding="utf-8")))
    cmap = {r["method"]: float(r["iou"]) for r in crit}
    methods = ["Random", r"$\ell_1$*", r"$I_{\mathrm{task}}$", r"$I_{\mathrm{contrast}}$", r"$I$ (full)"]
    ious = [
        cmap["Random"],
        cmap["L1"],
        cmap["I_task"],
        cmap["I_contrast"],
        cmap["I_full_main"],
    ]
    teacher = cmap["Teacher"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.bar(range(len(ious)), ious, width=0.72, color=(0.20, 0.45, 0.70), edgecolor="none")
    ax.axhline(teacher, ls="--", color=(0.75, 0.20, 0.20), lw=1.6, label=f"Teacher IoU = {teacher:.4f}")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("IoU")
    ax.set_ylim(0.68, 0.78)
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.set_title("Importance criterion comparison (DNA-Net / NUAA-SIRST)")
    ax.legend(loc="upper left", fontsize=9)
    for i, v in enumerate(ious):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = ROOT / "images" / "Matlab" / "敲除校准曲线" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig_criterion_bar"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    copy_out(stem, "fig_criterion_bar")
    plt.close(fig)
    print("criterion bar saved")


def plot_distill() -> None:
    dpath = ROOT / "images" / "Matlab" / "敲除校准曲线" / "data" / "distill_ablation.csv"
    dmap = {r["setting"]: r for r in csv.DictReader(dpath.open(encoding="utf-8"))}
    order = ["A1_no_retrain", "A2_seg_only", "A3_seg_region", "A4_full_main"]
    labels = ["No retrain", "Seg only", "Seg+Region", "Seg+Region+KD"]
    nuaa = [float(dmap[k]["nuaa_iou"]) for k in order]
    # NUDT KD 行保留消融设定 0.9184（λ_kd=0.5），NUAA 与主表对齐 0.7387
    nudt = [float(dmap[k]["nudt_iou"]) for k in order]
    nudt[-1] = float(dmap["A4_full_ablation"]["nudt_iou"])
    teacher_n = float(dmap["Teacher"]["nuaa_iou"])
    teacher_d = float(dmap["Teacher"]["nudt_iou"])

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=160)
    ax.bar(x - width / 2, nuaa, width, color=(0.20, 0.45, 0.70), label="NUAA-SIRST")
    ax.bar(x + width / 2, nudt, width, color=(0.85, 0.55, 0.20), label="NUDT-SIRST")
    ax.axhline(teacher_n, ls="--", color=(0.20, 0.45, 0.70), lw=1.2, label=f"Teacher NUAA={teacher_n:.4f}")
    ax.axhline(teacher_d, ls="-.", color=(0.85, 0.55, 0.20), lw=1.2, label=f"Teacher NUDT={teacher_d:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.set_title("Distillation ablations (DNA-Net)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = ROOT / "images" / "Matlab" / "敲除校准曲线" / "figures"
    stem = out / "fig_distill_bar"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    copy_out(stem, "fig_distill_bar")
    plt.close(fig)
    print("distill bar saved")


if __name__ == "__main__":
    fix_qualitative_legend()
    plot_pareto()
    plot_criterion()
    plot_distill()
    print("done")
