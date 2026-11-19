# -*- coding: utf-8 -*-
"""把 fig_qualitative 用到的 6 张样本整理进 images/定性对照/。

带框面板请用 make_boxed_panels_from_assets.py 从原图生成，不要从拼图裁切。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(r"d:\Desktop\红外检测模型\面向边缘部署的红外弱小目标检测可迁移压缩框架\images")
OUT = ROOT / "定性对照"
FIG = ROOT / "fig_qualitative.png"

# 与正文 fig_qualitative 从上到下一致
FIG6 = [
    ("NUAA-SIRST", "Misc_423"),
    ("NUAA-SIRST", "Misc_129"),
    ("SIRST-V2", "S20210422_S7_1"),
    ("SIRST-V2", "Misc_38"),
    ("NUAA-SIRST", "Misc_229"),
    ("SIRST-V2", "foo-00112"),
]
NEEDED = {"Input.png", "GT.png", "Teacher.png", "Student.png", "Input_RGB.png"}


def folder_name(ds: str, iid: str) -> str:
    return f"{ds}_{iid}"


def move_extras_to_backup() -> None:
    backup = OUT / "备选_未入正文图"
    backup.mkdir(parents=True, exist_ok=True)
    keep = {folder_name(ds, iid) for ds, iid in FIG6}
    for p in OUT.iterdir():
        if not p.is_dir():
            continue
        if p.name in keep or p.name.startswith("备选"):
            continue
        dest = backup / p.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(p), str(dest))
        print(f"[move] {p.name} -> 备选_未入正文图/")


def assert_assets() -> None:
    missing = []
    for ds, iid in FIG6:
        d = OUT / folder_name(ds, iid)
        if not d.is_dir():
            missing.append(f"{d} (folder)")
            continue
        for name in NEEDED:
            if not (d / name).is_file():
                missing.append(f"{d / name}")
    if missing:
        raise SystemExit("缺少素材，请先跑 export_qualitative_assets.py:\n" + "\n".join(missing))


def crop_panels_from_figure() -> None:
    """从拼图裁切每行 Input/GT/Teacher/Student，便于对照正文排版。"""
    im = Image.open(FIG).convert("RGB")
    w, h = im.size
    # 经验几何：顶栏约 28px，底图例约 28px，左侧标签约 118px，四列等分
    top = 28
    bottom = h - 28
    left = 118
    usable_h = bottom - top
    usable_w = w - left
    row_h = usable_h / 6.0
    col_w = usable_w / 4.0
    cols = ["Input", "GT", "Teacher", "Student"]

    for r, (ds, iid) in enumerate(FIG6):
        d = OUT / folder_name(ds, iid)
        d.mkdir(parents=True, exist_ok=True)
        y0 = int(round(top + r * row_h))
        y1 = int(round(top + (r + 1) * row_h))
        for c, cname in enumerate(cols):
            x0 = int(round(left + c * col_w))
            x1 = int(round(left + (c + 1) * col_w))
            crop = im.crop((x0, y0, x1, y1))
            out = d / f"Panel_{cname}.png"
            crop.save(out)
        print(f"[crop] {d.name}/Panel_*.png")


def write_readme() -> None:
    lines = [
        "正文 fig_qualitative 用到的素材（按图中从上到下）",
        "路径: images/定性对照/<数据集>_<样本ID>/",
        "",
        "每个子文件夹包含:",
        "  Input.png / Input_RGB.png / GT.png / Teacher.png / Student.png",
        "    -> 无画框原图/掩膜，便于你自己排版",
        "  Panel_Input.png / Panel_GT.png / Panel_Teacher.png / Panel_Student.png",
        "    -> 从正文拼图裁出的对应格子（含框与局部放大，仅供对照）",
        "",
        "正文 6 张:",
    ]
    for i, (ds, iid) in enumerate(FIG6, 1):
        lines.append(f"  {i}. {folder_name(ds, iid)}")
    lines += [
        "",
        "未进入正文拼图的样本在: 备选_未入正文图/",
    ]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[save]", OUT / "README.txt")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    assert_assets()
    move_extras_to_backup()
    crop_panels_from_figure()
    write_readme()
    print("[done]", OUT)
