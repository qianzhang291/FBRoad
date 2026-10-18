from pathlib import Path
import cv2
import numpy as np
import torch
from .data import prepare_input
from .metrics import restore_probability
from .runtime import load_checkpoint
from .factory import make_region_model

class RoadExtractor:
    def __init__(self, region_checkpoint, boundary_checkpoint=None, arm='ema',
                 seed=23456, device='cpu'):
        self.device = torch.device(device)
        self.region = make_region_model(arm,seed)
        self.region.load_state_dict(load_checkpoint(region_checkpoint)['model'],strict=True)
        self.region.to(self.device).eval().requires_grad_(False)
        self.boundary = None
        if boundary_checkpoint:
            from .boundary import make_model
            self.boundary = make_model()
            self.boundary.load_state_dict(load_checkpoint(boundary_checkpoint)['model'],strict=True)
            self.boundary.to(self.device).eval().requires_grad_(False)
    @torch.inference_mode()
    def predict(self, image_bgr, gps):
        p = self.region(prepare_input(image_bgr,gps).unsqueeze(0).to(self.device))
        result = restore_probability(p[0,0].cpu().numpy(),image_bgr.shape[:2]) > .5
        if self.boundary is not None:
            from .boundary import normalize_bgr, refine
            # Configure cuDNN backend flags for boundary refinement.
            with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                            deterministic=False, allow_tf32=True):
                b,d = self.boundary(normalize_bgr(image_bgr).unsqueeze(0).to(self.device))
                result,_ = refine(result,b,d)
        return result
