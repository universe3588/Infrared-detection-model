import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(pred, target, weight=None, smooth=1.0):
    # SCTrans 输出已是概率时不要二次 sigmoid
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    if weight is None:
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1.0 - (intersection + smooth) / (union + smooth)
    w = weight
    intersection = (pred * target * w).sum()
    union = (pred * w).sum() + (target * w).sum() - intersection
    return 1.0 - (intersection + smooth) / (union + smooth)


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
        return soft_iou_loss(pred, target, weight=weight)


def kd_prob_loss(student_prob, teacher_prob):
    """两端已是 [0,1] 概率（SCTrans forward 已 sigmoid）。"""
    return F.mse_loss(student_prob, teacher_prob.detach())


def take_final(preds):
    if isinstance(preds, (list, tuple)):
        return preds[-1]
    return preds


def multi_bce_loss(bce, preds, labels):
    """与教师 Net.loss 对齐：tuple 多头各自 BCE 再求和。"""
    if isinstance(preds, (list, tuple)):
        losses = [bce(p, labels) for p in preds]
        total = sum(losses)
        return total, total / max(1, len(losses))
    return bce(preds, labels), bce(preds, labels)
