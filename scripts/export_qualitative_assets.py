#!/usr/bin/env python3
"""导出论文定性图用到的 Input/GT/Teacher/Student 原图到 images/定性对照/<数据集>_<编号>/"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DNANET = os.path.join(WORKSPACE, "DNANet")
DATA_ROOT = os.path.join(WORKSPACE, "RDANet-main", "datasets")
OUT_ROOT = os.path.join(
    WORKSPACE, "面向边缘部署的红外弱小目标检测可迁移压缩框架", "images", "定性对照"
)
SEL = os.path.join(
    WORKSPACE,
    "面向边缘部署的红外弱小目标检测可迁移压缩框架",
    "images",
    "qualitative_selection.csv",
)
# 当前正文 fig_qualitative 使用的 6 张（2+2 success + 2 fail）
FIG6_IDS = {
    ("NUAA-SIRST", "Misc_423"),
    ("NUAA-SIRST", "Misc_129"),
    ("SIRST-V2", "S20210422_S7_1"),
    ("SIRST-V2", "Misc_38"),
    ("NUAA-SIRST", "Misc_229"),
    ("SIRST-V2", "foo-00112"),
}
CKPTS = {
    "NUAA-SIRST": (
        os.path.join(DNANET, "result", "NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS", "mIoU__DNANet_NUAA-SIRST_epoch.pth.tar"),
        os.path.join(DNANET, "compress_runs", "NUAA-SIRST_pipeline_20261029_181406", "student_best.pth.tar"),
    ),
    "SIRST-V2": (
        os.path.join(DNANET, "result", "SIRST-V2_DNANet_14_10_2026_23_22_42_wDS", "mIoU__DNANet_SIRST-V2_epoch.pth.tar"),
        os.path.join(DNANET, "compress_runs", "SIRST-V2_pipeline_20261029_200429", "student_best.pth.tar"),
    ),
}
BASE, THR = 256, 0.5


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state(path):
    ckpt = torch_load(path)
    for k in ("state_dict", "model", "net", "student", "teacher"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt, ckpt[k]
    return ckpt, ckpt


@torch.no_grad()
def predict(model, tensor, device):
    from compress.losses import take_final
    x = tensor.unsqueeze(0).to(device)
    pred = torch.sigmoid(take_final(model(x)))[0, 0].cpu().numpy()
    return (pred >= THR).astype(np.uint8)


def build_models(ds, device):
    sys.path.insert(0, DNANET)
    os.chdir(DNANET)
    from model.load_param_data import load_param
    from model.model_DNANet import DNANet, Res_CBAM_block

    tck, sck = CKPTS[ds]
    nb, blocks = load_param("three", "resnet_18")
    teacher = DNANet(1, 3, Res_CBAM_block, list(blocks), list(nb), True).to(device).eval()
    _, st = load_state(tck)
    teacher.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    ckpt, ss = load_state(sck)
    student = DNANet(1, 3, Res_CBAM_block, list(blocks), list(ckpt["nb_filter"]), True).to(device).eval()
    student.load_state_dict({k.replace("module.", ""): v for k, v in ss.items()}, strict=False)
    for p in list(teacher.parameters()) + list(student.parameters()):
        p.requires_grad_(False)
    return teacher, student


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # also export extras from csv for用户可选
    items = []
    with open(SEL, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append((row["dataset"], row["id"], row["role"]))

    # ensure fig6 set first
    ordered = []
    seen = set()
    for ds, iid in [
        ("NUAA-SIRST", "Misc_423"),
        ("NUAA-SIRST", "Misc_129"),
        ("SIRST-V2", "S20210422_S7_1"),
        ("SIRST-V2", "Misc_38"),
        ("NUAA-SIRST", "Misc_229"),
        ("SIRST-V2", "foo-00112"),
    ]:
        ordered.append((ds, iid, "fig6"))
        seen.add((ds, iid))
    for ds, iid, role in items:
        if (ds, iid) not in seen:
            ordered.append((ds, iid, role))
            seen.add((ds, iid))

    os.makedirs(OUT_ROOT, exist_ok=True)
    readme = os.path.join(OUT_ROOT, "README.txt")
    models = {}
    lines = [
        "论文定性图素材（无画框、无拼图，便于自行排版）",
        "目录: <数据集>_<样本ID>/{Input,GT,Teacher,Student}.png",
        "分辨率: 256x256；Teacher/Student 为 thr=0.5 二值掩膜",
        "fig6 标记 = 当前正文 fig_qualitative 使用的 6 张",
        "",
    ]

    for ds, iid, tag in ordered:
        if ds not in models:
            models[ds] = build_models(ds, device)
            print(f"[model] {ds}")
        teacher, student = models[ds]
        img_path = os.path.join(DATA_ROOT, ds, "images", iid + ".png")
        mask_path = os.path.join(DATA_ROOT, ds, "masks", iid + ".png")
        if not os.path.isfile(img_path):
            print(f"[skip] missing {img_path}")
            continue
        raw = Image.open(img_path).convert("RGB").resize((BASE, BASE), Image.BILINEAR)
        gt = Image.open(mask_path).convert("L").resize((BASE, BASE), Image.NEAREST)
        gt_np = (np.array(gt) > 127).astype(np.uint8)
        tensor = tf(raw)
        t_bin = predict(teacher, tensor, device)
        s_bin = predict(student, tensor, device)

        folder = os.path.join(OUT_ROOT, f"{ds}_{iid}")
        os.makedirs(folder, exist_ok=True)
        # Input: save grayscale IR for layout friendliness + RGB
        gray = raw.convert("L")
        gray.save(os.path.join(folder, "Input.png"))
        raw.save(os.path.join(folder, "Input_RGB.png"))
        Image.fromarray(gt_np * 255).save(os.path.join(folder, "GT.png"))
        Image.fromarray(t_bin * 255).save(os.path.join(folder, "Teacher.png"))
        Image.fromarray(s_bin * 255).save(os.path.join(folder, "Student.png"))
        lines.append(f"{ds}_{iid}\t{tag}")
        print(f"[save] {folder}")

    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] {readme}")


if __name__ == "__main__":
    main()
