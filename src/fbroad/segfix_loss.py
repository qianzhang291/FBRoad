"""Author-style class-balanced boundary and eight-direction losses."""
import torch
from torch.nn import functional as F


def balanced_ce(logits, labels, classes, eligible=None):
    if eligible is None:
        eligible = torch.ones_like(labels, dtype=torch.bool)
    if not bool(eligible.any()):
        return logits.sum() * 0
    selected = labels[eligible].long()
    counts = torch.bincount(selected, minlength=classes).float()
    weights = (1. - counts / counts.sum()).clamp_min(1e-3)
    losses = F.cross_entropy(logits, labels.long(), weight=weights, reduction='none')
    return losses[eligible].sum() / weights[selected].sum().clamp_min(1e-6)


def segfix_loss(boundary_logits, direction_logits, boundary_target, direction_target):
    size = boundary_target.shape[-2:]
    boundary_logits = F.interpolate(boundary_logits.float(), size=size,
                                    mode='bilinear', align_corners=True)
    direction_logits = F.interpolate(direction_logits.float(), size=size,
                                     mode='bilinear', align_corners=True)
    mask_loss = balanced_ce(boundary_logits, boundary_target, 2)
    # BJRoad has only road/background; supervise inward direction at the true
    # boundary during learning so the direction head receives a stable signal.
    # Author code further intersects with its current boundary prediction.
    direction_loss = balanced_ce(direction_logits, direction_target, 8,
                                 boundary_target.bool())
    total = mask_loss + direction_loss
    return total, {'boundary': mask_loss.detach(),
                   'direction': direction_loss.detach()}
