"""
MSHNet transferable compression pipeline.

Modes: score | prune | distill | pipeline

Example:
  python compress_train.py --mode pipeline \\
    --teacher_ckpt weight/MSHNet-NUAA-SIRST-*/weight.pkl \\
    --dataset NUAA-SIRST --dataset_dir ../RDANet-main/datasets --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adagrad, Adam
from torch.optim import lr_scheduler

from compress.adapters.mshnet_adapter import TEACHER_CHANNELS, MSHNetAdapter
from compress.importance import ImportanceEstimator, choose_keep_indices
from compress.losses import (
    RegionSoftIoULoss,
    kd_logits_loss,
    soft_iou_loss,
    take_final,
    take_sides,
)
from compress.progress import StageTimer, make_pbar
from compress.prune import (
    build_pruned_mshnet,
    count_parameters,
    load_keep_masks,
    save_keep_masks,
    student_channel_config,
)
from model.MSHNet import MSHNet
from utils.data import IRSTD_Dataset
from utils.metric import mIoU as mIoUMetric


def parse_args():
    p = argparse.ArgumentParser(description="MSHNet compression pipeline")
    p.add_argument("--mode", type=str, default="pipeline",
                   choices=["score", "prune", "distill", "pipeline"])
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, default="")
    p.add_argument("--scores_path", type=str, default="")
    p.add_argument("--keep_path", type=str, default="")
    p.add_argument("--work_dir", type=str, default="./compress_runs")

    p.add_argument("--dataset", type=str, default="NUAA-SIRST",
                   choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "SIRST-V2"])
    p.add_argument("--dataset_dir", type=str, default="../RDANet-main/datasets")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--base_size", type=int, default=256)
    p.add_argument("--crop_size", type=int, default=256)

    p.add_argument("--score_batches", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--prune_rates", type=str, default="0.20,0.30,0.40,0.50,0.60",
                   help="per-stage prune ratios for widths 16/32/64/128/256")
    p.add_argument("--min_keep", type=int, default=4)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, default="Adagrad", choices=["Adagrad", "Adam"])
    p.add_argument("--warm_epoch", type=int, default=5,
                   help="distill 内多尺度 warm；默认与教师训练一致")
    p.add_argument("--lambda_seg", type=float, default=1.0)
    p.add_argument("--lambda_kd", type=float, default=1.0)
    p.add_argument("--lambda_region", type=float, default=0.5)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_state_dict_flexible(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "net", "model", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], ckpt
        sample = next(iter(ckpt.values()))
        if torch.is_tensor(sample):
            return ckpt, {}
    raise ValueError(f"Cannot parse state_dict from {ckpt_path}")


def build_loaders(args):
    data_args = SimpleNamespace(
        dataset_dir=os.path.join(args.dataset_dir, args.dataset),
        crop_size=args.crop_size,
        base_size=args.base_size,
    )
    trainset = IRSTD_Dataset(data_args, mode="train")
    valset = IRSTD_Dataset(data_args, mode="val")
    train_loader = Data.DataLoader(
        trainset, args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = Data.DataLoader(valset, 1, shuffle=False, drop_last=False, num_workers=0)
    print(f"[data] {args.dataset} train={len(trainset)} val={len(valset)}")
    return train_loader, val_loader


def build_teacher(ckpt_path, device):
    model = MSHNet(3)
    state, meta = load_state_dict_flexible(ckpt_path)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:6], "..." if len(missing) > 6 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:6], "..." if len(unexpected) > 6 else "")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[teacher] params={count_parameters(model):.3f}M")
    return model


@torch.no_grad()
def evaluate(model, loader, device, desc: str = "[eval]"):
    model.eval()
    metric = mIoUMetric(1)
    total = len(loader) if hasattr(loader, "__len__") else None
    pbar = make_pbar(loader, total=total, desc=desc, unit="img", leave=False)
    for data, mask in pbar:
        data = data.to(device)
        mask = mask.to(device)
        _, pred = model(data, True)
        metric.update(pred, mask)
        _, mean_iou = metric.get()
        pbar.set_postfix(iou=f"{mean_iou:.4f}")
    model.train()
    _, mean_iou = metric.get()
    return float(mean_iou)


def run_score(args, teacher, train_loader, device, work_dir):
    timer = StageTimer("P1 score")
    adapter = MSHNetAdapter(TEACHER_CHANNELS)
    for p in teacher.parameters():
        p.requires_grad_(True)
    estimator = ImportanceEstimator(
        adapter, alpha=args.alpha, dilate=args.dilate, max_batches=args.score_batches,
    )
    scores = estimator.estimate(teacher, train_loader, device=device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    scores_path = args.scores_path or os.path.join(work_dir, "scores.pt")
    torch.save({"scores": scores, "channels": list(TEACHER_CHANNELS)}, scores_path)
    print(f"[save] scores -> {scores_path}")

    rates = [float(x) for x in args.prune_rates.split(",")]
    assert len(rates) == 5, "MSHNet needs 5 prune_rates for channels 16/32/64/128/256"
    keep_idx = choose_keep_indices(scores, rates, min_keep=args.min_keep)
    keep_path = args.keep_path or os.path.join(work_dir, "keep_masks.pt")
    sw = student_channel_config(keep_idx)
    save_keep_masks(keep_path, keep_idx, meta={
        "channels_teacher": list(TEACHER_CHANNELS),
        "channels_student": sw,
        "prune_rates": rates,
        "alpha": args.alpha,
        "dataset": args.dataset,
    })
    timer.done(f"student channels={sw}")
    return scores_path, keep_path, keep_idx


def run_prune(args, teacher, keep_idx, device, work_dir):
    timer = StageTimer("P2 prune")
    student = build_pruned_mshnet(teacher, keep_idx=keep_idx, input_channels=3)
    student.to(device)
    sw = student_channel_config(keep_idx)
    print(f"[student] params={count_parameters(student):.3f}M")
    out = os.path.join(work_dir, "student_init.pth.tar")
    torch.save({
        "state_dict": student.state_dict(),
        "channels": sw,
        "param_channels": sw,
    }, out)
    print(f"[save] student init -> {out}")
    timer.done(f"params={count_parameters(student):.3f}M")
    return student, out


def multi_scale_seg_loss(sides, final, labels, down):
    """Align with MSHNet teacher: SoftIoU on final + progressively downsampled sides."""
    loss = soft_iou_loss(final, labels)
    cur = labels
    for j, side in enumerate(sides):
        if j > 0:
            cur = down(cur)
        if side.shape[-2:] != cur.shape[-2:]:
            cur = torch.nn.functional.interpolate(cur, size=side.shape[-2:], mode="nearest")
        loss = loss + soft_iou_loss(side, cur)
    return loss / max(1, len(sides) + 1)


def run_distill(args, teacher, student, train_loader, val_loader, device, work_dir):
    timer = StageTimer("P3 distill")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    if args.optimizer == "Adam":
        optimizer = Adam(student.parameters(), lr=args.lr)
    else:
        optimizer = Adagrad(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    region_crit = RegionSoftIoULoss(dilate=args.dilate)
    down = nn.MaxPool2d(2, 2)

    best_iou = -1.0
    log_path = os.path.join(work_dir, "distill_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,test_iou,seg,region,kd\n")

    n_train = len(train_loader) if hasattr(train_loader, "__len__") else None
    epoch_bar = make_pbar(total=args.epochs, desc=f"[P3] {args.dataset}", unit="epoch", leave=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        warm = epoch > args.warm_epoch
        run_loss = run_seg = run_reg = run_kd = 0.0
        n = 0
        t_ep = time.time()
        batch_bar = make_pbar(
            train_loader, total=n_train,
            desc=f"  train {epoch}/{args.epochs}", unit="batch", leave=False,
        )
        for data, labels in batch_bar:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            s_out = student(data, warm)
            with torch.no_grad():
                t_out = teacher(data, warm)

            s_final = take_final(s_out)
            t_final = take_final(t_out)
            sides = take_sides(s_out)
            loss_seg = multi_scale_seg_loss(sides, s_final, labels, down)
            loss_region = region_crit(s_final, labels)
            loss_kd = kd_logits_loss(s_final, t_final)
            loss = (
                args.lambda_seg * loss_seg
                + args.lambda_region * loss_region
                + args.lambda_kd * loss_kd
            )
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())
            run_seg += float(loss_seg.item())
            run_reg += float(loss_region.item())
            run_kd += float(loss_kd.item())
            n += 1
            batch_bar.set_postfix(
                loss=f"{run_loss / n:.4f}",
                kd=f"{run_kd / n:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        batch_bar.close()
        scheduler.step()

        avg = run_loss / max(1, n)
        test_iou = -1.0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_iou = evaluate(
                student, val_loader, device, desc=f"  eval  {epoch}/{args.epochs}",
            )
            if test_iou > best_iou:
                best_iou = test_iou
                path = os.path.join(work_dir, "student_best.pth.tar")
                torch.save({
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "best_iou": best_iou,
                    "channels": list(student.param_channels),
                    "param_channels": list(student.param_channels),
                }, path)
                epoch_bar.write(f"[save] best IoU={best_iou:.4f} -> {path}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg:.6f},{test_iou:.6f},"
                f"{run_seg/max(1,n):.6f},{run_reg/max(1,n):.6f},{run_kd/max(1,n):.6f}\n"
            )
        ep_sec = time.time() - t_ep
        epoch_bar.set_postfix(
            loss=f"{avg:.4f}",
            iou=f"{test_iou:.4f}" if test_iou >= 0 else "n/a",
            best=f"{best_iou:.4f}" if best_iou >= 0 else "n/a",
            ep_s=f"{ep_sec:.0f}",
        )
        epoch_bar.update(1)

    epoch_bar.close()
    timer.done(f"best_iou={best_iou:.4f}")
    return best_iou


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(args.work_dir, f"{args.dataset}_{args.mode}_{stamp}")
    if args.mode == "pipeline":
        work_dir = os.path.join(args.work_dir, f"{args.dataset}_pipeline_{stamp}")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[work_dir] {work_dir}")

    train_loader, val_loader = build_loaders(args)
    teacher = build_teacher(args.teacher_ckpt, device)

    keep_idx = None
    student = None

    if args.mode in ("score", "pipeline"):
        _, keep_path, keep_idx = run_score(args, teacher, train_loader, device, work_dir)
        args.keep_path = keep_path

    if args.mode in ("prune", "pipeline"):
        if keep_idx is None:
            keep_idx, _ = load_keep_masks(args.keep_path)
        student, _ = run_prune(args, teacher, keep_idx, device, work_dir)

    if args.mode in ("distill", "pipeline"):
        if student is None:
            if not args.student_ckpt:
                raise SystemExit("distill needs --student_ckpt or pipeline/prune first")
            ckpt = torch.load(args.student_ckpt, map_location="cpu")
            ch = ckpt.get("channels") or ckpt.get("param_channels") or list(TEACHER_CHANNELS)
            student = MSHNet(3, channels=list(ch))
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            student.load_state_dict(state, strict=False)
            student.to(device)
            if keep_idx is None and args.keep_path:
                keep_idx, _ = load_keep_masks(args.keep_path)
        run_distill(args, teacher, student, train_loader, val_loader, device, work_dir)

    print(f"[done] artifacts in {work_dir}")


if __name__ == "__main__":
    main()
