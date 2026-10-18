"""Author SegFix_HRNet and fixed binary-road offset application."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ._vendor.segfix.lib.utils.tools.configer import Configer
from ._vendor.segfix.lib.models.nets.segfix import SegFix_HRNet



MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
VECTORS = np.array([
    [0, -1], [-1, -1], [-1, 0], [-1, 1],
    [0, 1], [1, 1], [1, 0], [1, -1],
], dtype=np.int16)


def make_model(pretrained=None):
    config = Configer(config_dict={'network': {
        'backbone': 'hrnet2x20', 'pretrained': None, 'bn_type': 'torchbn'}})
    model = SegFix_HRNet(config)
    if pretrained:
        weights = torch.load(pretrained, map_location='cpu', weights_only=False)
        missing, extra = model.backbone.load_state_dict(weights, strict=False)
        assert not missing and len(extra) == 157, (missing[:5], extra[:5])
        assert all(key.startswith(('incre_modules.', 'downsamp_modules.', 'final_layer.',
                                   'classifier.'))
                   for key in extra), extra[:10]
    return model


def normalize_bgr(image):
    assert image.ndim == 3 and image.shape[2] == 3
    return torch.from_numpy(((image.transpose(2, 0, 1).astype(np.float32) / 255.
                             - MEAN) / STD).copy())


def refine(binary_prediction, boundary_logits, direction_logits):
    """All-pixel author-style 8-way offsets and factor-two bilinear label shift."""
    assert binary_prediction.ndim == 2
    height, width = binary_prediction.shape
    assert boundary_logits.shape[0] == direction_logits.shape[0] == 1
    with torch.no_grad():
        b = F.interpolate(boundary_logits.float(), size=(height, width),
                          mode='bilinear', align_corners=True)
        d = F.interpolate(direction_logits.float(), size=(height, width),
                          mode='bilinear', align_corners=True)
        edge = (b.softmax(1)[0, 1] > .5).cpu().numpy()
        direction = d.argmax(1)[0].cpu().numpy()
    offset = VECTORS[direction]
    yy, xx = np.indices((height, width), dtype=np.float32)
    # Original semantic-label refinement script uses --scale 2 and bilinear
    # sampling before rounding. Unselected pixels receive zero offset.
    yy += 2. * edge * offset[:, :, 0]
    xx += 2. * edge * offset[:, :, 1]
    shifted = cv2.remap(binary_prediction.astype(np.float32), xx, yy,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    result = np.rint(shifted).astype(bool)
    assert result.shape == binary_prediction.shape
    return result, {'boundary_pixels': int(edge.sum()),
                    'changed_pixels': int((result != binary_prediction).sum())}
