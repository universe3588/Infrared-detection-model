"""
IRSTD transferable compression pipeline on DNANet.

Modes:
  score   : load teacher -> compute stage importance -> save scores/keep masks
  prune   : load teacher + keep masks -> build student -> save student init ckpt
  distill : distill/finetune student from teacher
  pipeline: score -> prune -> distill
  eval    : fair re-eval teacher (+ optional student) with the SAME evaluate()/loader

Example (after a trained teacher exists):
  python compress_train.py --mode pipeline --teacher_ckpt path/to/teacher.pth.tar \\
      --dataset IRSTD-1k --root ../RDANet-main/datasets --epochs 200

Fair re-eval (解释学生暴涨):
  python compress_train.py --mode eval \\
      --teacher_ckpt result/.../mIoU__DNANet_*.pth.tar \\
      --student_ckpt compress_runs/.../student_best.pth.tar \\
      --dataset NUAA-SIRST --root ../RDANet-main/datasets
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from compress.adapters.dnanet_adapter import DNANetAdapter
from compress.importance import ImportanceEstimator, choose_keep_indices
from compress.losses import (
    RegionSoftIoULoss,
    kd_logits_loss,
    prediction_loss,
    take_final,
)
from compress.prune import (
    build_pruned_dnanet,
    count_parameters,
    load_keep_masks,
    save_keep_masks,
)
from model.load_param_data import load_dataset, load_param
from model.loss import SoftIoULoss
from model.metric import ROCMetric, mIoU
from model.model_DNANet import DNANet, Res_CBAM_block
from model.utils import AverageMeter, TestSetLoader, TrainSetLoader


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser(description="DNANet compression pipeline")
    p.add_argument("--mode", type=str, default="pipeline",
                   choices=["score", "prune", "distill", "pipeline", "eval"])
    p.add_argument("--teacher_ckpt", type=str, required=True,
                   help="teacher checkpoint (.pth.tar), state_dict or {'state_dict':...}")
    p.add_argument("--student_ckpt", type=str, default="",
                   help="optional student ckpt (distill init, or eval fair compare)")
    p.add_argument("--scores_path", type=str, default="",
                   help="path to save/load importance scores .pt")
    p.add_argument("--keep_path", type=str, default="",
                   help="path to save/load keep masks .pt")
    p.add_argument("--work_dir", type=str, default="./compress_runs")

    # data / model (aligned with train.py)
    p.add_argument("--dataset", type=str, default="IRSTD-1k")
    p.add_argument("--root", type=str, default="../RDANet-main/datasets")
    p.add_argument("--split_method", type=str, default="50_50")
    p.add_argument("--suffix", type=str, default=".png")
    p.add_argument("--subset_size", type=int, default=0)
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--base_size", type=int, default=256)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--channel_size", type=str, default="three")
    p.add_argument("--backbone", type=str, default="resnet_18")
    p.add_argument("--deep_supervision", type=str, default="True")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--test_batch_size", type=int, default=4)
    p.add_argument(
        "--train_aug", type=str, default="resize", choices=["resize", "full"],
        help="resize: only resize/normalize (推荐蒸馏，避免小目标被随机裁掉); "
             "full: DNANet 原版随机缩放裁剪增强",
    )

    # scoring / pruning
    p.add_argument("--score_batches", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--prune_rates", type=str, default="0.20,0.30,0.40,0.50,0.60",
                   help="per-stage prune ratios stage0..stage4")
    p.add_argument("--min_keep", type=int, default=4)

    # distill train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, default="Adagrad")
    p.add_argument("--lambda_seg", type=float, default=1.0)
    p.add_argument("--lambda_kd", type=float, default=1.0)
    p.add_argument("--lambda_region", type=float, default=0.5,
                   help="extra region SoftIoU weight on student")
    p.add_argument("--gpus", type=str, default="0")
    return p.parse_args()


def build_loaders(args):
    dataset_dir = os.path.join(args.root, args.dataset)
    train_ids, val_ids, _ = load_dataset(
        args.root, args.dataset, args.split_method,
        subset_size=args.subset_size, subset_seed=args.subset_seed,
    )
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # NUDT 等小目标集上，TrainSetLoader 的随机缩放+裁剪很容易裁掉目标；
    # SoftIoU 在空标签上会把网络推向“全背景”，蒸馏表现为 test_loss≈0.996、mIoU=0。
    if args.train_aug == "full":
        trainset = TrainSetLoader(
            dataset_dir, img_id=train_ids, base_size=args.base_size,
            crop_size=args.crop_size, transform=tfm, suffix=args.suffix,
        )
    else:
        trainset = TestSetLoader(
            dataset_dir, img_id=train_ids, base_size=args.base_size,
            crop_size=args.crop_size, transform=tfm, suffix=args.suffix,
        )
    testset = TestSetLoader(
        dataset_dir, img_id=val_ids, base_size=args.base_size,
        crop_size=args.crop_size, transform=tfm, suffix=args.suffix,
    )
    train_loader = DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )
    test_loader = DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False,
    )
    return train_loader, test_loader


def load_state_dict_flexible(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # maybe raw state dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            # heuristic: looks like state_dict
            sample = next(iter(ckpt.values()))
            if torch.is_tensor(sample):
                return ckpt
    raise ValueError(f"Cannot parse state_dict from {ckpt_path}")


def build_teacher(args, device):
    nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
    deep_sup = args.deep_supervision  # keep string behavior consistent with train.py
    model = DNANet(
        num_classes=1,
        input_channels=args.in_channels,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=deep_sup,
    )
    state = load_state_dict_flexible(args.teacher_ckpt)
    # strip module. prefix if any
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[teacher] params={count_parameters(model):.3f}M  nb_filter={nb_filter}")
    return model, nb_filter, num_blocks


@torch.no_grad()
def evaluate(model, loader, device, deep_supervision=True):
    model.eval()
    metric = mIoU(1)
    losses = AverageMeter()
    use_ds = deep_supervision == "True" or deep_supervision is True
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        preds = model(data)
        pred = take_final(preds)
        loss = SoftIoULoss(pred, labels)
        losses.update(loss.item(), pred.size(0))
        metric.update(pred, labels)
    pixacc, miou = metric.get()
    return losses.avg, float(miou), float(pixacc)


def build_student_from_ckpt(ckpt_path, args, device, num_blocks):
    """Rebuild narrowed DNANet from student_best.pth.tar (has nb_filter + state_dict)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"student ckpt must contain state_dict: {ckpt_path}")
    nb_filter = ckpt.get("nb_filter")
    if not nb_filter:
        raise ValueError(f"student ckpt missing nb_filter: {ckpt_path}")
    deep_sup = args.deep_supervision
    model = DNANet(
        num_classes=1,
        input_channels=args.in_channels,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=list(nb_filter),
        deep_supervision=deep_sup,
    )
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] student missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[warn] student unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logged_best = ckpt.get("best_iou", None)
    print(
        f"[student] params={count_parameters(model):.3f}M  nb_filter={list(nb_filter)}"
        + (f"  ckpt_best_iou={float(logged_best):.4f}" if logged_best is not None else "")
    )
    return model, list(nb_filter), logged_best


def run_fair_eval(args, teacher, test_loader, device, work_dir, num_blocks):
    """Same evaluate() for teacher and optional student; write JSON for paper defense."""
    t_loss, t_iou, t_pix = evaluate(teacher, test_loader, device, args.deep_supervision)
    print(f"[fair-eval][teacher] loss={t_loss:.4f} mIoU={t_iou:.4f} pixAcc={t_pix:.4f}")

    result = {
        "dataset": args.dataset,
        "teacher_ckpt": args.teacher_ckpt,
        "student_ckpt": args.student_ckpt or None,
        "teacher_loss": t_loss,
        "teacher_mIoU": t_iou,
        "teacher_pixAcc": t_pix,
        "student_loss": None,
        "student_mIoU": None,
        "student_pixAcc": None,
        "student_ckpt_best_iou": None,
        "delta_pp": None,
        "note": "teacher/student evaluated with identical TestSetLoader + SoftIoU/mIoU in compress_train.evaluate",
    }

    if args.student_ckpt:
        student, _, logged_best = build_student_from_ckpt(
            args.student_ckpt, args, device, num_blocks,
        )
        s_loss, s_iou, s_pix = evaluate(student, test_loader, device, args.deep_supervision)
        print(f"[fair-eval][student] loss={s_loss:.4f} mIoU={s_iou:.4f} pixAcc={s_pix:.4f}")
        result["student_loss"] = s_loss
        result["student_mIoU"] = s_iou
        result["student_pixAcc"] = s_pix
        result["student_ckpt_best_iou"] = float(logged_best) if logged_best is not None else None
        result["delta_pp"] = (s_iou - t_iou) * 100.0
        print(
            f"[fair-eval][delta] student-teacher = {result['delta_pp']:+.2f} pp "
            f"(student {s_iou:.4f} vs teacher {t_iou:.4f})"
        )

    out_json = os.path.join(work_dir, f"{args.dataset}_eval_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    # also flat copy under --work_dir root for overnight summary scripts
    flat_json = os.path.join(args.work_dir, f"{args.dataset}_eval_metrics.json")
    if os.path.abspath(flat_json) != os.path.abspath(out_json):
        os.makedirs(args.work_dir, exist_ok=True)
        with open(flat_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[save] fair eval (flat) -> {flat_json}")
    print(f"[save] fair eval -> {out_json}")
    return result


def run_score(args, teacher, nb_filter, train_loader, device, work_dir):
    adapter = DNANetAdapter(nb_filter)
    # enable grads temporarily for weight.grad scoring
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
    torch.save({"scores": scores, "nb_filter": nb_filter}, scores_path)
    print(f"[save] scores -> {scores_path}")

    rates = [float(x) for x in args.prune_rates.split(",")]
    assert len(rates) == len(nb_filter), "prune_rates must match number of stages"
    keep_idx = choose_keep_indices(scores, rates, min_keep=args.min_keep)
    keep_path = args.keep_path or os.path.join(work_dir, "keep_masks.pt")
    save_keep_masks(keep_path, keep_idx, meta={
        "nb_filter": nb_filter,
        "prune_rates": rates,
        "alpha": args.alpha,
        "dataset": args.dataset,
    })
    return scores_path, keep_path, keep_idx


def run_prune(args, teacher, num_blocks, keep_idx, device, work_dir):
    student = build_pruned_dnanet(
        teacher,
        keep_idx=keep_idx,
        num_blocks=num_blocks,
        input_channels=args.in_channels,
        deep_supervision=args.deep_supervision,
        block=Res_CBAM_block,
    )
    student.to(device)
    print(f"[student] params={count_parameters(student):.3f}M")
    out = os.path.join(work_dir, "student_init.pth.tar")
    torch.save({
        "state_dict": student.state_dict(),
        "nb_filter": [int(keep_idx[s].numel()) for s in range(5)],
        "num_blocks": list(num_blocks),
        "deep_supervision": args.deep_supervision,
    }, out)
    print(f"[save] student init -> {out}")
    return student, out


def run_distill(args, teacher, student, train_loader, test_loader, device, work_dir):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(student.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adagrad(student.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    seg_crit = SoftIoULoss
    region_crit = RegionSoftIoULoss(dilate=args.dilate)
    use_ds = args.deep_supervision == "True"

    best_iou = 0.0
    log_path = os.path.join(work_dir, "distill_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,test_loss,miou\n")

    for epoch in range(args.epochs):
        student.train()
        losses = AverageMeter()
        tbar = tqdm(train_loader, desc=f"distill {epoch}")
        for data, labels in tbar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = student(data)
            with torch.no_grad():
                t_out = teacher(data)

            loss_seg = prediction_loss(s_out, labels, seg_crit, deep_supervision=use_ds)
            loss_region = prediction_loss(s_out, labels, region_crit, deep_supervision=use_ds)
            loss_kd = kd_logits_loss(take_final(s_out), take_final(t_out))
            loss = (
                args.lambda_seg * loss_seg
                + args.lambda_region * loss_region
                + args.lambda_kd * loss_kd
            )
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.size(0))
            tbar.set_description(f"Epoch {epoch} loss {losses.avg:.4f}")

        scheduler.step()
        test_loss, miou, _pix = evaluate(student, test_loader, device, args.deep_supervision)
        print(f"[eval] epoch={epoch} test_loss={test_loss:.4f} mIoU={miou:.4f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{losses.avg:.6f},{test_loss:.6f},{miou:.6f}\n")

        if miou > best_iou:
            best_iou = miou
            ckpt = os.path.join(work_dir, "student_best.pth.tar")
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "best_iou": best_iou,
                "nb_filter": [
                    student.conv0_0[0].conv2.out_channels,
                    student.conv1_0[0].conv2.out_channels,
                    student.conv2_0[0].conv2.out_channels,
                    student.conv3_0[0].conv2.out_channels,
                    student.conv4_0[0].conv2.out_channels,
                ],
            }, ckpt)
            print(f"[save] best mIoU={best_iou:.4f} -> {ckpt}")

    return best_iou


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(args.work_dir, f"{args.dataset}_{args.mode}_{stamp}")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[work_dir] {work_dir}")

    train_loader, test_loader = build_loaders(args)
    teacher, nb_filter, num_blocks = build_teacher(args, device)

    if args.mode == "eval":
        run_fair_eval(args, teacher, test_loader, device, work_dir, num_blocks)
        return

    t_loss, t_iou, t_pix = evaluate(teacher, test_loader, device, args.deep_supervision)
    print(f"[teacher eval] loss={t_loss:.4f} mIoU={t_iou:.4f} pixAcc={t_pix:.4f}")

    keep_idx = None
    student = None

    if args.mode in ("score", "pipeline"):
        _, keep_path, keep_idx = run_score(
            args, teacher, nb_filter, train_loader, device, work_dir,
        )
    elif args.keep_path:
        keep_idx, _ = load_keep_masks(args.keep_path)

    if args.mode in ("prune", "pipeline"):
        if keep_idx is None:
            raise ValueError("keep masks required: run score first or pass --keep_path")
        student, _ = run_prune(args, teacher, num_blocks, keep_idx, device, work_dir)
        s_loss, s_iou, s_pix = evaluate(student, test_loader, device, args.deep_supervision)
        print(f"[student init eval] loss={s_loss:.4f} mIoU={s_iou:.4f} pixAcc={s_pix:.4f}")

    if args.mode in ("distill", "pipeline"):
        if student is None:
            if not args.student_ckpt:
                raise ValueError("distill needs --student_ckpt or prune/pipeline mode")
            # rebuild from keep masks + load weights
            if keep_idx is None:
                raise ValueError("pass --keep_path with student_ckpt")
            student, _ = run_prune(args, teacher, num_blocks, keep_idx, device, work_dir)
            state = load_state_dict_flexible(args.student_ckpt)
            student.load_state_dict(state, strict=False)
            student.to(device)
        best = run_distill(args, teacher, student, train_loader, test_loader, device, work_dir)
        print(f"[done] best student mIoU={best:.4f}")


if __name__ == "__main__":
    main()
