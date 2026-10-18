"""BJRoad binary-label adaptation of author SegFix distance/direction targets."""
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset

from .boundary import normalize_bgr


def _sobel_kernel(axis):
    kernel = np.zeros((5, 5), dtype=np.float32)
    for y in range(5):
        for x in range(5):
            dy, dx = y - 2, x - 2
            if dy or dx:
                kernel[y, x] = (dx if axis == 0 else dy) / (dx * dx + dy * dy)
    return kernel


KX, KY = _sobel_kernel(0), _sobel_kernel(1)


def targets_from_mask(binary):
    """Both valid BJRoad classes get interior distance and inward direction."""
    assert binary.dtype == bool and binary.shape == (1024, 1024)
    depth = np.zeros(binary.shape, dtype=np.float32)
    dy = np.zeros_like(depth)
    dx = np.zeros_like(depth)
    for inside in (binary, ~binary):
        distance = distance_transform_edt(inside).astype(np.float32)
        depth[inside] = distance[inside]
        gy = cv2.filter2D(distance, -1, KY, borderType=cv2.BORDER_CONSTANT)
        gx = cv2.filter2D(distance, -1, KX, borderType=cv2.BORDER_CONSTANT)
        dy[inside] = gy[inside]
        dx[inside] = gx[inside]
    angle = np.degrees(np.arctan2(dy, dx))
    direction = (np.floor(((angle + 180. + 22.5) % 360.) / 45.) % 8).astype(np.uint8)
    boundary = (depth <= 5.).astype(np.uint8)
    return np.stack((boundary, direction), -1)


class TrainDataset(Dataset):
    def __init__(self, split, epoch, data_root, target_root, seed=23456):
        self.ids = split['train']
        self.source = Path(data_root) / 'train_val' / 'image'
        self.targets = Path(target_root)
        self.epoch = epoch
        self.seed = seed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        image = cv2.imread(str(self.source / f'{sample}_sat.png'))
        target = np.load(self.targets / f'{sample}.npy')
        assert image.shape == (1024, 1024, 3) and target.shape == (1024, 1024, 2)
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index * 104729)
        top, left = (int(rng.integers(0, 513)) for _ in range(2))
        image = image[top:top + 512, left:left + 512].copy()
        target = target[top:top + 512, left:left + 512].copy()
        if rng.random() < .5:
            image = image[:, ::-1].copy()
            target = target[:, ::-1].copy()
            target[:, :, 1] = (4 - target[:, :, 1]) % 8
        brightness = int(rng.integers(-10, 11))
        image = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        return normalize_bgr(image), torch.from_numpy(target[:, :, 0].astype(np.int64)), \
            torch.from_numpy(target[:, :, 1].astype(np.int64)), sample
