#!/usr/bin/env python3
"""
Ex04 / AutoDL 用 L2SKNet 训练脚本（基于官方 train_device0.py）。

相对官方改动：
  - 支持 NUAA-SIRST / SIRST-V2（RDANet 划分）
  - --device 可选；默认 cuda:0
  - 默认只存 last + best_miou（避免 400 epoch 占满磁盘）
  - 每 epoch 评测；日志写到 log/

示例（AutoDL）：
  python prepare_data.py --rda-root /root/autodl-tmp/datasets
  python train_ex04.py --model_names L2SKNet_UNet --dataset_names NUAA-SIRST --batchSize 8
  python train_ex04.py --model_names L2SKNet_UNet --dataset_names NUAA-SIRST NUDT-SIRST IRSTD-1K SIRST-V2 --batchSize 4
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evaluation.mIoU import mIoU
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP
from net import Net
from utils.datasets import build_dataset
from utils.utils import get_optimizer, seed_pytorch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch L2SKNet train (Ex04 / AutoDL)")
parser.add_argument(
    "--model_names",
    default=["L2SKNet_UNet"],
    type=str,
    nargs="+",
    help="L2SKNet_UNet | L2SKNet_FPN | L2SKNet_1D_UNet | L2SKNet_1D_FPN",
)
parser.add_argument(
    "--dataset_names",
    default=["NUAA-SIRST"],
    type=str,
    nargs="+",
    help="NUAA-SIRST NUDT-SIRST IRSTD-1K SIRST-V2 SIRST-aug",
)
parser.add_argument("--dataset_dir", default="./data", type=str)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--save", default="./log", type=str)
parser.add_argument("--nEpochs", type=int, default=400)
parser.add_argument("--optimizer_name", default="Adam", type=str)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--save_every", type=int, default=0, help=">0 时额外按间隔存 epoch ckpt；0=只存 last/best")
parser.add_argument("--resume", default="", type=str, help="optional .pth.tar to resume")

opt = parser.parse_args()
seed_pytorch(opt.seed)


def _dataset_base(name: str) -> str:
    return os.path.join(opt.dataset_dir, name) + os.sep


def train_one():
    device = torch.device(opt.device if torch.cuda.is_available() or "cpu" in opt.device else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; use --device cpu or install CUDA torch")

    train_set = build_dataset(opt.dataset_name, base_dir=_dataset_base(opt.dataset_name), mode="trainval")
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
        pin_memory=str(device).startswith("cuda"),
    )

    net = Net(model_name=opt.model_name).to(device)
    net.train()

    epoch_state = 0
    opt.best_miou = 0.0
    opt.best_miou_epoch = 0
    opt.best_fscore = 0.0
    opt.best_fscore_epoch = 0
    opt.best_pd = 0.0
    opt.best_pd_epoch = 0
    opt.best_fa = 1.0
    opt.best_fa_epoch = 0
    total_loss_list = []

    if opt.resume and os.path.isfile(opt.resume):
        ckpt = torch.load(opt.resume, map_location="cpu")
        net.load_state_dict(ckpt["state_dict"])
        epoch_state = int(ckpt.get("epoch", 0))
        total_loss_list = ckpt.get("total_loss", [])
        print(f"[resume] {opt.resume} @ epoch {epoch_state}")

    if opt.optimizer_name == "Adam":
        optimizer_settings = {"lr": 5e-4}
        scheduler_name = "MultiStepLR"
        scheduler_settings = {"epochs": opt.nEpochs, "step": [200, 300], "gamma": 0.1}
    elif opt.optimizer_name == "Adagrad":
        optimizer_settings = {"lr": 0.05}
        scheduler_name = "CosineAnnealingLR"
        scheduler_settings = {"epochs": 1500, "min_lr": 1e-3}
        opt.nEpochs = 1500
    else:
        raise NotImplementedError(opt.optimizer_name)

    # get_optimizer 读 opt.step / opt.gamma 等（兼容官方 utils）
    opt.step = scheduler_settings.get("step", [200, 300])
    opt.gamma = scheduler_settings.get("gamma", 0.1)
    if "min_lr" in scheduler_settings:
        opt.min_lr = scheduler_settings["min_lr"]

    optimizer, scheduler = get_optimizer(
        net, opt.optimizer_name, scheduler_name, optimizer_settings, scheduler_settings
    )

    ckpt_dir = os.path.join(opt.save, opt.dataset_name, opt.model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for idx_epoch in range(epoch_state, opt.nEpochs):
        net.train()
        total_loss_epoch = []
        for img, gt_mask in train_loader:
            if img.shape[0] == 1:
                continue
            img = Variable(img).to(device)
            gt_mask = Variable(gt_mask).to(device)
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        mean_loss = float(np.array(total_loss_epoch).mean()) if total_loss_epoch else 0.0
        total_loss_list.append(mean_loss)
        print(time.ctime()[4:-5] + " Epoch---%d, total_loss---%f," % (idx_epoch + 1, mean_loss))
        opt.f.write(time.ctime()[4:-5] + " Epoch---%d, total_loss---%f,\n" % (idx_epoch + 1, mean_loss))
        opt.f.flush()

        state = {
            "epoch": idx_epoch + 1,
            "state_dict": net.state_dict(),
            "total_loss": total_loss_list,
            "best_miou": opt.best_miou,
        }
        last_pth = os.path.join(ckpt_dir, "last.pth.tar")
        torch.save(state, last_pth)

        if opt.save_every > 0 and ((idx_epoch + 1) % opt.save_every == 0):
            torch.save(state, os.path.join(ckpt_dir, "%d.pth.tar" % (idx_epoch + 1)))

        test_and_maybe_save_best(net, device, ckpt_dir, idx_epoch, total_loss_list)


def test_and_maybe_save_best(net, device, ckpt_dir, idx_epoch, total_loss_list):
    test_set = build_dataset(opt.dataset_name, base_dir=_dataset_base(opt.dataset_name), mode="test")
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net.eval()
    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for img, gt_mask, size, _ in test_loader:
        with torch.no_grad():
            img = Variable(img).to(device)
            pred = net.forward(img)
            pred = pred[:, :, : size[0], : size[1]]
        gt_mask = gt_mask[:, :, : size[0], : size[1]]
        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update(
            pred[0, 0, :, :].cpu().detach().numpy(),
            gt_mask[0, 0, :, :].detach().numpy(),
            size,
        )
        eval_mIoU_P_R_F.update(
            labels=gt_mask[0, 0, :, :].detach().numpy(),
            preds=pred[0, 0, :, :].cpu().detach().numpy(),
        )

    pixAcc, Ying_mIoU = eval_mIoU.get()
    pd, fa = eval_PD_FA.get()
    _, _, _, fscore = eval_mIoU_P_R_F.get()

    if Ying_mIoU > opt.best_miou:
        opt.best_miou = Ying_mIoU
        opt.best_miou_epoch = idx_epoch + 1
        torch.save(
            {
                "epoch": idx_epoch + 1,
                "state_dict": net.state_dict(),
                "total_loss": total_loss_list,
                "best_miou": opt.best_miou,
                "pd": pd,
                "fa": fa,
                "fscore": fscore,
            },
            os.path.join(ckpt_dir, "best_miou.pth.tar"),
        )

    if fscore > opt.best_fscore:
        opt.best_fscore = fscore
        opt.best_fscore_epoch = idx_epoch + 1
    if pd > opt.best_pd:
        opt.best_pd = pd
        opt.best_pd_epoch = idx_epoch + 1
    if fa < opt.best_fa:
        opt.best_fa = fa
        opt.best_fa_epoch = idx_epoch + 1

    msg = (
        "pixAcc %.6f, mIoU: %.6f\n"
        "Pd: %.6f, Fa: %.8f, fscore: %.6f\n"
        "Best mIoU: %.6f @ Epoch=%d | Best fscore: %.6f @ Epoch=%d\n"
        "Best Pd: %.6f @ Epoch=%d | Best Fa: %.8f @ Epoch=%d\n"
        % (
            pixAcc,
            Ying_mIoU,
            pd,
            fa,
            fscore,
            opt.best_miou,
            opt.best_miou_epoch,
            opt.best_fscore,
            opt.best_fscore_epoch,
            opt.best_pd,
            opt.best_pd_epoch,
            opt.best_fa,
            opt.best_fa_epoch,
        )
    )
    print(msg)
    opt.f.write(msg)
    opt.f.flush()


if __name__ == "__main__":
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            os.makedirs(opt.save, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(opt.save, "%s_%s_%s.txt" % (dataset_name, model_name, stamp))
            opt.f = open(log_path, "w", encoding="utf-8")
            print(dataset_name + "\t" + model_name)
            print("log ->", log_path)
            train_one()
            print("\n")
            opt.f.close()
