import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int):
    probs = torch.softmax(logits, dim=1)
    targets = targets.clone()

    valid_mask = targets != ignore_index
    targets = targets.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)
    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return 1.0 - dice.mean()


def build_loss(cfg, num_classes: int, ignore_index: int):
    loss_type = cfg.model.loss.type
    dice_weight = float(cfg.model.loss.dice_weight)
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    if loss_type == "ce":
        return ce_loss

    if loss_type == "ce_dice":
        def _loss(logits, targets):
            return ce_loss(logits, targets) + dice_weight * dice_loss(
                logits, targets, num_classes=num_classes, ignore_index=ignore_index
            )

        return _loss

    raise ValueError(f"Unknown loss type: {loss_type}")
