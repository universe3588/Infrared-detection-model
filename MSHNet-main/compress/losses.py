import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(pred, target, weight=None, smooth=1.0):
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


def kd_logits_loss(student_logits, teacher_logits, T=1.0):
    s = torch.sigmoid(student_logits / T)
    t = torch.sigmoid(teacher_logits / T).detach()
    return F.mse_loss(s, t)


def take_final(preds):
    """MSHNet returns (side_list, final) or a bare tensor."""
    if isinstance(preds, (list, tuple)):
        if len(preds) == 2 and isinstance(preds[0], (list, tuple)):
            return preds[1]
        return preds[-1]
    return preds


def take_sides(preds):
    if isinstance(preds, (list, tuple)) and len(preds) == 2 and isinstance(preds[0], (list, tuple)):
        return list(preds[0])
    return []
