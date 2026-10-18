"""DICN pixel contrast algorithm, reimplemented with declared device/label guards.

Source: BubbleSai/DICN a949a22f0ee381c90075c14092c97dcc933bda91,
loss.py PixelContrastLoss and networks/transformer/transformer.py ProjectionHead.
The author's outside-band prediction=-1 rule is preserved, including its actual
sampling consequence: ground-truth grouping can still sample outside the band.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=64, proj_dim=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim_in, dim_in, 1),
                                  nn.BatchNorm2d(dim_in), nn.ReLU(),
                                  nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


def boundary_band(labels):
    # OpenCV 8x8 morphology has anchor (4,4); asymmetric padding is deliberate.
    dilated = F.max_pool2d(F.pad(labels, (4, 3, 4, 3), value=0), 8, 1)
    eroded = -F.max_pool2d(F.pad(-labels, (4, 3, 4, 3), value=-1), 8, 1)
    return dilated != eroded


class PixelContrastLoss(nn.Module):
    temperature = .1
    base_temperature = .07
    max_samples = 1024
    max_views = 100

    def sample(self, features, truth, predicted, generator):
        groups = []
        for index in range(truth.shape[0]):
            for cls in (0, 1):
                eligible = truth[index] == cls
                if int(eligible.sum()) > self.max_views:
                    groups.append((index, cls, eligible))
        if not groups:
            return None, None, {'active': True, 'eligible_groups': 0, 'anchors': 0}
        views = min(self.max_samples // len(groups), self.max_views)
        sampled, classes, hard_total, easy_total = [], [], 0, 0
        for index, cls, eligible in groups:
            hard = (eligible & (predicted[index] != cls)).nonzero().flatten()
            easy = (eligible & (predicted[index] == cls)).nonzero().flatten()
            nh, ne = hard.numel(), easy.numel()
            if nh >= views / 2 and ne >= views / 2:
                keep_h = views // 2
                keep_e = views - keep_h
            elif nh >= views / 2:
                keep_e, keep_h = ne, views - ne
            else:
                keep_h, keep_e = nh, views - nh
            assert keep_h <= nh and keep_e <= ne
            hp = torch.randperm(nh, generator=generator)[:keep_h]
            ep = torch.randperm(ne, generator=generator)[:keep_e]
            indices = torch.cat((hard[hp], easy[ep]))
            sampled.append(features[index, indices])
            classes.append(cls)
            hard_total += keep_h
            easy_total += keep_e
        return torch.stack(sampled), torch.tensor(classes, device=features.device), {
            'active': True, 'eligible_groups': len(groups), 'views_per_group': views,
            'anchors': len(groups) * views, 'hard': hard_total, 'easy': easy_total,
            'road_groups': classes.count(1), 'background_groups': classes.count(0)}

    def contrast(self, sampled, classes):
        # View-major ordering and author denominator (one positive + all negatives).
        groups, views, _ = sampled.shape
        vectors = torch.cat(torch.unbind(sampled, dim=1), dim=0)
        logits = vectors @ vectors.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        same = (classes[:, None] == classes[None, :]).repeat(views, views)
        positive = same & ~torch.eye(groups * views, device=sampled.device, dtype=torch.bool)
        negatives = (~same).to(logits.dtype) * logits.exp()
        denominator = logits.exp() + negatives.sum(dim=1, keepdim=True)
        log_prob = logits - denominator.clamp_min(torch.finfo(logits.dtype).tiny).log()
        average = (positive * log_prob).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return -(self.temperature / self.base_temperature) * average.mean()

    def forward(self, embedding, target, prediction, seed):
        assert embedding is not None
        # Only this new auxiliary uses categorical nearest-neighbor labels.
        # Existing BCE+batchDice retains its original bilinear soft labels.
        truth = (F.interpolate(target, size=embedding.shape[-2:], mode='nearest') > .5).long()
        pred = (prediction.detach() > .5).long()
        assert pred.shape == truth.shape
        pred = pred.masked_fill(~boundary_band(truth.float()), -1)
        features = embedding.permute(0, 2, 3, 1).reshape(embedding.shape[0], -1, embedding.shape[1])
        generator = torch.Generator(device='cpu').manual_seed(seed)
        sampled, classes, info = self.sample(features, truth.flatten(1), pred.flatten(1), generator)
        if sampled is None:
            return embedding.flatten()[0] * 0, info
        return self.contrast(sampled, classes), info
