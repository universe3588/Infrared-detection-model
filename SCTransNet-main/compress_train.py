"""
SCTransNet transferable compression pipeline (DNANet-aligned).

Modes: score | prune | distill | pipeline

Example:
  python compress_train.py --mode pipeline \\
    --teacher_ckpt log/NUAA-SIRST/SCTransNet_781_best.pth.tar \\
    --dataset NUAA-SIRST --dataset_dir ../RDANet-main/datasets --epochs 350 \\
    --prune_rates 0.05,0.15,0.25,0.35 --lr 3e-4 --lambda_region 0.75
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Adagrad
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from compress.adapters.sctrans_adapter import TEACHER_CHANNELS, SCTransAdapter
from compress.importance import ImportanceEstimator, choose_keep_indices
from compress.losses import (
    RegionSoftIoULoss,
    kd_prob_loss,
    multi_bce_loss,
    take_final,
)
from compress.progress import StageTimer, make_pbar
from compress.prune import (
    build_pruned_sctrans,
    count_parameters,
    load_keep_masks,
    save_keep_masks,
    student_channel_config,
)
from dataset import TrainSetLoader, TestSetLoader
from metrics import mIoU as mIoUMetric
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet


def parse_args():
    p = argparse.ArgumentParser(description="SCTransNet compression pipeline")
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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--threads", type=int, default=0)

    p.add_argument("--score_batches", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--dilate", type=int, default=3)
    # SCTrans 定稿默认 = v3：Transformer 不剪时 CNN 配额更保守。
    # 协议仍是 P1→P2→P3；DNA/UIU/MSH 不受影响。
    p.add_argument("--prune_rates", type=str, default="0.05,0.15,0.25,0.35",
                   help="4 rates for channels 32/64/128/256 (v3 milder budget)")
    p.add_argument("--min_keep", type=int, default=4)

    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "Adagrad"])
    p.add_argument("--lambda_seg", type=float, default=1.0)
    p.add_argument("--lambda_kd", type=float, default=1.0)
    p.add_argument("--lambda_region", type=float, default=0.75)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def strip_state_dict(state):
    out = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def load_state_dict_flexible(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "net", "model", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return strip_state_dict(ckpt[key]), ckpt
        sample = next(iter(ckpt.values()))
        if torch.is_tensor(sample):
            return strip_state_dict(ckpt), {}
    raise ValueError(f"Cannot parse state_dict from {ckpt_path}")


def build_loaders(args):
    train_set = TrainSetLoader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset,
        patch_size=args.patch_size,
    )
    test_set = TestSetLoader(
        args.dataset_dir, args.dataset, args.dataset, img_norm_cfg=None,
    )
    train_loader = DataLoader(
        train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)
    print(f"[data] {args.dataset} train={len(train_set)} test={len(test_set)}")
    return train_loader, test_loader


def build_teacher(ckpt_path, device, img_size=256):
    cfg = get_SCTrans_config()
    model = SCTransNet(cfg, mode="train", deepsuper=True, img_size=img_size, channel_num=list(TEACHER_CHANNELS))
    state, meta = load_state_dict_flexible(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:6], "..." if len(missing) > 6 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:6], "..." if len(unexpected) > 6 else "")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[teacher] params={count_parameters(model):.3f}M  channels={model.channel_num}")
    return model


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, desc="[eval]"):
    model.eval()
    # temporarily test-mode style: use last head
    old_mode = model.mode
    model.mode = "test"
    metric = mIoUMetric()
    total = len(loader) if hasattr(loader, "__len__") else None
    pbar = make_pbar(loader, total=total, desc=desc, unit="img", leave=False)
    for batch in pbar:
        if len(batch) >= 3:
            img, gt_mask, size = batch[0], batch[1], batch[2]
        else:
            img, gt_mask = batch[0], batch[1]
            size = None
        img = img.to(device)
        pred = model(img)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        if size is not None:
            pred = pred[:, :, : size[0], : size[1]]
            gt_mask = gt_mask[:, :, : size[0], : size[1]]
        metric.update((pred > threshold).cpu(), gt_mask.cpu())
        res = metric.get()
        pbar.set_postfix(iou=f"{res[1]:.4f}")
    model.mode = old_mode
    model.train()
    return float(metric.get()[1])


def run_score(args, teacher, train_loader, device, work_dir):
    timer = StageTimer("P1 score")
    adapter = SCTransAdapter(TEACHER_CHANNELS)
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
    assert len(rates) == 4, "SCTransNet needs 4 prune_rates"
    keep_idx = choose_keep_indices(scores, rates, min_keep=args.min_keep, force_even=True)
    keep_path = args.keep_path or os.path.join(work_dir, "keep_masks.pt")
    sw = student_channel_config(keep_idx)
    save_keep_masks(keep_path, keep_idx, meta={
        "channels_teacher": list(TEACHER_CHANNELS),
        "channel_num_student": sw,
        "prune_rates": rates,
        "alpha": args.alpha,
        "dataset": args.dataset,
    })
    timer.done(f"student channel_num={sw}")
    return scores_path, keep_path, keep_idx


def run_prune(args, teacher, keep_idx, device, work_dir):
    timer = StageTimer("P2 prune")
    student = build_pruned_sctrans(
        teacher, keep_idx=keep_idx, img_size=args.patch_size, mode="train",
    )
    student.to(device)
    sw = student_channel_config(keep_idx)
    print(f"[student] params={count_parameters(student):.3f}M")
    out = os.path.join(work_dir, "student_init.pth.tar")
    torch.save({
        "state_dict": student.state_dict(),
        "channel_num": sw,
        "channels": sw,
    }, out)
    print(f"[save] student init -> {out}")
    timer.done(f"params={count_parameters(student):.3f}M")
    return student, out


def run_distill(args, teacher, student, train_loader, test_loader, device, work_dir):
    timer = StageTimer("P3 distill")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    student.mode = "train"
    for p in student.parameters():
        p.requires_grad_(True)

    if args.optimizer == "Adagrad":
        optimizer = Adagrad(student.parameters(), lr=args.lr)
    else:
        optimizer = Adam(student.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    bce = nn.BCELoss(reduction="mean")
    region_crit = RegionSoftIoULoss(dilate=args.dilate)

    best_iou = -1.0
    log_path = os.path.join(work_dir, "distill_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,test_iou,seg,region,kd\n")

    n_train = len(train_loader) if hasattr(train_loader, "__len__") else None
    epoch_bar = make_pbar(total=args.epochs, desc=f"[P3] {args.dataset}", unit="epoch", leave=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        student.mode = "train"
        run_loss = run_seg = run_reg = run_kd = 0.0
        n = 0
        t_ep = time.time()
        batch_bar = make_pbar(
            train_loader, total=n_train,
            desc=f"  train {epoch}/{args.epochs}", unit="batch", leave=False,
        )
        for img, gt_mask in batch_bar:
            if img.shape[0] == 1:
                continue
            img = img.to(device)
            gt_mask = gt_mask.to(device)
            optimizer.zero_grad()
            s_out = student(img)
            with torch.no_grad():
                t_out = teacher(img)

            _, loss_seg = multi_bce_loss(bce, s_out, gt_mask)
            s_final = take_final(s_out)
            t_final = take_final(t_out)
            loss_region = region_crit(s_final, gt_mask)
            loss_kd = kd_prob_loss(s_final, t_final)
            loss = (
                args.lambda_seg * loss_seg
                + args.lambda_region * loss_region
                + args.lambda_kd * loss_kd
            )
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())
            run_seg += float(loss_seg.item() if torch.is_tensor(loss_seg) else loss_seg)
            run_reg += float(loss_region.item())
            run_kd += float(loss_kd.item())
            n += 1
            batch_bar.set_postfix(
                loss=f"{run_loss / max(1,n):.4f}",
                kd=f"{run_kd / max(1,n):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        batch_bar.close()
        scheduler.step()

        avg = run_loss / max(1, n)
        test_iou = -1.0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_iou = evaluate(
                student, test_loader, device,
                threshold=args.threshold,
                desc=f"  eval  {epoch}/{args.epochs}",
            )
            if test_iou > best_iou:
                best_iou = test_iou
                path = os.path.join(work_dir, "student_best.pth.tar")
                torch.save({
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "best_iou": best_iou,
                    "channel_num": list(student.channel_num),
                    "channels": list(student.channel_num),
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

    train_loader, test_loader = build_loaders(args)
    teacher = build_teacher(args.teacher_ckpt, device, img_size=args.patch_size)

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
            ch = ckpt.get("channel_num") or ckpt.get("channels") or list(TEACHER_CHANNELS)
            cfg = get_SCTrans_config()
            student = SCTransNet(
                cfg, mode="train", deepsuper=True,
                img_size=args.patch_size, channel_num=list(ch),
            )
            state = strip_state_dict(ckpt.get("state_dict", ckpt))
            student.load_state_dict(state, strict=False)
            student.to(device)
        run_distill(args, teacher, student, train_loader, test_loader, device, work_dir)

    print(f"[done] artifacts in {work_dir}")


if __name__ == "__main__":
    main()
