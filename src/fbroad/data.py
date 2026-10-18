from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ._vendor.cmmpnet.utils.data_loader import ImageGPSDataset

class RegionDataset(Dataset):
    def __init__(self, ids, data_root, part, seed, epoch=0, augment=False):
        self.ids, self.seed, self.epoch, self.augment = ids, seed, epoch, augment
        root = Path(data_root) / part
        self.inner = ImageGPSDataset(ids, str(root/'image'), str(root/'mask'),
                                     str(root/'gps'), randomize=augment, down_scale=True)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        if self.augment:
            np.random.seed((self.seed + self.epoch * 1000003 + index * 104729) % 2**32)
        x, y = self.inner[index]
        if x.shape != (4,512,512) or y.shape != (1,1024,1024):
            raise ValueError('Expected native 1024x1024 BJRoad samples')
        return x, y, self.ids[index]

def load_split(path):
    split = json.loads(Path(path).read_text())
    for key in ('train', 'validation', 'test'):
        if len(split[key]) != len(set(split[key])):
            raise ValueError(f'Duplicate IDs in {key}')
    def mothers(ids):
        return {(s.split('_')[0], int(s.split('_')[1]) % 100) for s in ids}
    if mothers(split['train']) & mothers(split['validation']):
        raise ValueError('Training and validation mother images overlap')
    return split

def region_loader(ids, data_root, part, seed, batch=8, epoch=0, train=False, workers=4):
    return DataLoader(RegionDataset(ids, data_root, part, seed, epoch, train),
        batch_size=batch, shuffle=train, generator=torch.Generator().manual_seed(seed+epoch),
        num_workers=workers, pin_memory=True, drop_last=False)

def prepare_input(image_bgr, gps):
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or gps.shape != image_bgr.shape[:2]:
        raise ValueError('Expected BGR image and spatially aligned grayscale GPS map')
    four = cv2.resize(np.concatenate((image_bgr, gps[:,:,None]), -1), (512,512))
    return torch.from_numpy((four.astype(np.float32).transpose(2,0,1)/255*3.2-1.6).copy())
