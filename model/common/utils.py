import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


def BCEDiceLoss(inputs, targets, eps=1e-6):
    """BCE-with-logits plus soft Dice; inputs/targets are [B, 1, H, W]."""
    bce = F.binary_cross_entropy_with_logits(inputs, targets)
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return bce + 1 - dice
