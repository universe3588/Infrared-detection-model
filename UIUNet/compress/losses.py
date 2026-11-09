"""Losses for UIU-Net compression. UIU heads already output probabilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_prob(pred, target, weight=None, smooth=1.0):
    """SoftIoU on probabilities in [0,1] (no extra sigmoid)."""
    pred = pred.clamp(0.0, 1.0)
    if weight is None:
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return 1.0 - (inter + smooth) / (union + smooth)
    w = weight
    inter = (pred * target * w).sum()
    union = (pred * w).sum() + (target * w).sum() - inter
    return 1.0 - (inter + smooth) / (union + smooth)


def make_target_region_weight(mask, dilate=3):
    if dilate <= 0:
        return torch.clamp(mask, 0.0, 1.0)
    k = 2 * dilate + 1
    pooled = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=dilate)
    region = (pooled > 0).float()
    return 0.05 + 0.95 * region


class RegionSoftIoULoss(nn.Module):
    def __init__(self, dilate=3):
        super().__init__()
        self.dilate = dilate

    def forward(self, pred, target):
        weight = make_target_region_weight(target, self.dilate)
        return soft_iou_prob(pred, target, weight=weight)


def kd_prob_loss(student_prob, teacher_prob):
    return F.mse_loss(student_prob.clamp(0, 1), teacher_prob.clamp(0, 1).detach())


def take_final(preds):
    """UIU returns (d0,d1,...,d6); fused head is d0."""
    if isinstance(preds, (list, tuple)):
        return preds[0]
    return preds


def multi_bce_loss(bce, preds, labels):
    losses = [bce(p, labels) for p in preds]
    return losses[0], sum(losses)
