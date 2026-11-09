"""
UIU-Net transferable compression pipeline.

Modes: score | prune | distill | pipeline

Example:
  python compress_train.py --mode pipeline \\
    --teacher_ckpt saved_models/uiunet_irstd/NUAA-SIRST_xxx/uiunet_best.pth.tar \\
    --dataset NUAA-SIRST --dataset_dir ../RDANet-main/datasets --epochs 200
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
import torch.optim as optim
from torch.optim import lr_scheduler

from compress.adapters.uiunet_adapter import TEACHER_STAGE_WIDTHS, UIUNetAdapter
from compress.importance import ImportanceEstimator, choose_keep_indices
from compress.losses import (
    RegionSoftIoULoss,
    kd_prob_loss,
    multi_bce_loss,
    take_final,
)
from compress.progress import StageTimer, make_pbar
from compress.prune import (
    build_pruned_uiunet,
    count_parameters,
    load_keep_masks,
    save_keep_masks,
    student_channel_config,
)
from model import UIUNET
from train_irstd import build_loaders, soft_iou


def parse_args():
    p = argparse.ArgumentParser(description="UIU-Net compression pipeline")
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
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rescale", type=int, default=320)
    p.add_argument("--crop", type=int, default=288)
    # distill 推荐用较小 crop 风险；默认沿用教师训练设定

    p.add_argument("--score_batches", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--prune_rates", type=str, default="0.20,0.30,0.40,0.50",
                   help="per-stage prune ratios for widths 64/128/256/512")
    p.add_argument("--min_keep", type=int, default=4)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "Adagrad"])
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
        for key in ("state_dict", "model", "net", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], ckpt
        sample = next(iter(ckpt.values()))
        if torch.is_tensor(sample):
            return ckpt, {}
    raise ValueError(f"Cannot parse state_dict from {ckpt_path}")


def build_teacher(ckpt_path, device):
    model = UIUNET(3, 1)
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
    best = meta.get("best_iou", None) if isinstance(meta, dict) else None
    print(f"[teacher] params={count_parameters(model):.3f}M  ckpt_best_iou={best}")
    return model


@torch.no_grad()
def evaluate(model, loader, device, desc: str = "[eval]"):
    model.eval()
    ious = []
    total = len(loader) if hasattr(loader, "__len__") else None
    pbar = make_pbar(loader, total=total, desc=desc, unit="img", leave=False)
    for data in pbar:
        inputs = data["image"].type(torch.FloatTensor).to(device)
        labels = data["label"].type(torch.FloatTensor).to(device)
        d0, *_ = model(inputs)
        iou = soft_iou(d0, labels)
        ious.append(iou)
        pbar.set_postfix(iou=f"{iou:.4f}")
    model.train()
    return float(np.mean(ious)) if ious else 0.0


def run_score(args, teacher, train_loader, device, work_dir):
    timer = StageTimer("P1 score")
    adapter = UIUNetAdapter(TEACHER_STAGE_WIDTHS)
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
    torch.save({"scores": scores, "stage_widths": list(TEACHER_STAGE_WIDTHS)}, scores_path)
    print(f"[save] scores -> {scores_path}")

    rates = [float(x) for x in args.prune_rates.split(",")]
    assert len(rates) == 4, "UIU-Net uses 4 stage widths; need 4 prune_rates"
    keep_idx = choose_keep_indices(scores, rates, min_keep=args.min_keep)
    keep_path = args.keep_path or os.path.join(work_dir, "keep_masks.pt")
    enc, dec, me, md, sw = student_channel_config(keep_idx, min_keep=args.min_keep)
    save_keep_masks(keep_path, keep_idx, meta={
        "stage_widths_teacher": list(TEACHER_STAGE_WIDTHS),
        "stage_widths_student": sw,
        "enc_outs": enc,
        "dec_outs": dec,
        "mids_enc": me,
        "mids_dec": md,
        "prune_rates": rates,
        "alpha": args.alpha,
        "dataset": args.dataset,
    })
    timer.done(f"student widths={sw}")
    return scores_path, keep_path, keep_idx


def run_prune(args, teacher, keep_idx, device, work_dir):
    timer = StageTimer("P2 prune")
    student = build_pruned_uiunet(
        teacher, keep_idx=keep_idx, min_keep=args.min_keep, in_ch=3, out_ch=1,
    )
    student.to(device)
    enc, dec, me, md, sw = student_channel_config(keep_idx, min_keep=args.min_keep)
    print(f"[student] params={count_parameters(student):.3f}M")
    out = os.path.join(work_dir, "student_init.pth.tar")
    torch.save({
        "state_dict": student.state_dict(),
        "enc_outs": enc,
        "dec_outs": dec,
        "mids_enc": me,
        "mids_dec": md,
        "stage_widths": sw,
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
    for p in student.parameters():
        p.requires_grad_(True)

    if args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(student.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(
            student.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
        )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    bce = nn.BCELoss(reduction="mean")
    region_crit = RegionSoftIoULoss(dilate=args.dilate)

    best_iou = -1.0
    log_path = os.path.join(work_dir, "distill_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,test_iou,seg,region,kd\n")

    n_train = len(train_loader) if hasattr(train_loader, "__len__") else None
    epoch_bar = make_pbar(
        total=args.epochs,
        desc=f"[P3] {args.dataset}",
        unit="epoch",
        leave=True,
    )

    for epoch in range(1, args.epochs + 1):
        student.train()
        run_loss = 0.0
        run_seg = run_reg = run_kd = 0.0
        n = 0
        t_ep = time.time()
        batch_bar = make_pbar(
            train_loader,
            total=n_train,
            desc=f"  train {epoch}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for data in batch_bar:
            inputs = data["image"].type(torch.FloatTensor).to(device)
            labels = data["label"].type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            s_out = student(inputs)
            with torch.no_grad():
                t_out = teacher(inputs)

            _, loss_seg = multi_bce_loss(bce, s_out, labels)
            loss_region = region_crit(take_final(s_out), labels)
            loss_kd = kd_prob_loss(take_final(s_out), take_final(t_out))
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
                student, test_loader, device,
                desc=f"  eval  {epoch}/{args.epochs}",
            )
            if test_iou > best_iou:
                best_iou = test_iou
                path = os.path.join(work_dir, "student_best.pth.tar")
                torch.save({
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "best_iou": best_iou,
                    "enc_outs": list(student.enc_outs),
                    "dec_outs": list(student.dec_outs),
                    "mids_enc": list(student.mids_enc),
                    "mids_dec": list(student.mids_dec),
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
            ep_s=f"{ep_sec:.0f}s",
        )
        epoch_bar.update(1)

    epoch_bar.close()
    timer.done(f"best_iou={best_iou:.4f}")
    return best_iou


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # shim: build_loaders expects args with dataset_dir fields already on namespace
    class _A:
        pass
    data_args = _A()
    data_args.dataset = args.dataset
    data_args.dataset_dir = args.dataset_dir
    data_args.batch_size = args.batch_size
    data_args.workers = args.workers
    data_args.rescale = args.rescale
    data_args.crop = args.crop

    root = os.path.join(args.dataset_dir, args.dataset)
    if not os.path.isdir(root):
        raise SystemExit(f"dataset root not found: {root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(args.work_dir, f"{args.dataset}_{args.mode}_{stamp}")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    pipeline_t0 = time.time()
    print(f"[work_dir] {work_dir}", flush=True)
    print(
        f"[plan] mode={args.mode} dataset={args.dataset} "
        f"epochs={args.epochs} score_batches={args.score_batches} "
        f"prune_rates={args.prune_rates}",
        flush=True,
    )

    train_loader, test_loader = build_loaders(data_args, root)
    teacher = build_teacher(args.teacher_ckpt, device)
    t_iou = evaluate(teacher, test_loader, device, desc="[teacher eval]")
    print(f"[teacher eval] test_IoU={t_iou:.4f}", flush=True)

    keep_idx = None
    student = None

    if args.mode in ("score", "pipeline"):
        _, _, keep_idx = run_score(args, teacher, train_loader, device, work_dir)
    elif args.keep_path:
        keep_idx, _ = load_keep_masks(args.keep_path)

    if args.mode in ("prune", "pipeline"):
        if keep_idx is None:
            raise ValueError("keep masks required")
        student, _ = run_prune(args, teacher, keep_idx, device, work_dir)
        s_iou = evaluate(student, test_loader, device, desc="[student init eval]")
        print(f"[student init eval] test_IoU={s_iou:.4f}", flush=True)

    if args.mode in ("distill", "pipeline"):
        if student is None:
            if keep_idx is None:
                raise ValueError("pass --keep_path")
            student, _ = run_prune(args, teacher, keep_idx, device, work_dir)
            if args.student_ckpt:
                state, _ = load_state_dict_flexible(args.student_ckpt)
                student.load_state_dict(state, strict=False)
                student.to(device)
        best = run_distill(args, teacher, student, train_loader, test_loader, device, work_dir)
        print(f"[done] best student IoU={best:.4f}", flush=True)

    total_sec = int(time.time() - pipeline_t0)
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    print(f"[pipeline] total wall time {h:02d}:{m:02d}:{s:02d}", flush=True)


if __name__ == "__main__":
    main()
