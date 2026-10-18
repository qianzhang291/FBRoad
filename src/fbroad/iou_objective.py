"""One fixed loss replacement, preserving the original BCE and soft-label resize.

The full MIT-licensed binary Lovasz hinge is in lovasz_losses.py, byte-identical
to bermanmaxim/LovaszSoftmax 7d48792d35a04d3167de488dd00daabbccd8334b.
Existing contrastive learning and model paths are unchanged.
"""
import torch
from torch.nn import functional as F
from .lovasz_losses import lovasz_hinge


def road_objective(criterion, target, prediction, logits, enabled):
    # Preserve the pinned author's OpenCV linear resize and soft labels for BCE.
    soft_target = target
    if target.shape[-2:] != prediction.shape[-2:]:
        soft_target = criterion.resize(target, *prediction.shape[-2:]).to(prediction.device)
    bce = criterion.bce_loss(prediction, soft_target)
    dice = criterion.soft_dice_loss(soft_target, prediction)
    original = bce + dice
    hinge = original.new_zeros(())
    if enabled:
        assert logits is not None and logits.shape == prediction.shape
        hard_target = (F.interpolate(target, size=prediction.shape[-2:], mode='nearest') > .5).long()
        # All batch foreground pixels participate. No void masks or test labels.
        hinge = lovasz_hinge(logits[:, 0].contiguous(), hard_target[:, 0].contiguous(),
                             per_image=False, ignore=None)
        primary = bce + hinge
    else:
        primary = original
    assert torch.isfinite(primary)
    return primary, {
        'enabled': enabled, 'bce': float(bce.detach()), 'original_dice': float(dice.detach()),
        'original_main': float(original.detach()), 'lovasz_hinge': float(hinge.detach()),
        'actual_main': float(primary.detach()), 'replacement_weight': 1. if enabled else 0.,
        'per_image': False}
