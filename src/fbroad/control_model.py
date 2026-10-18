"""Unchanged paired backbone with one training-only DICN-derived projector."""
import torch
from torch import nn
from .backbone import CMMPNetCMXDirection
from .contrast import ProjectionHead


class ContrastRoadNet(CMMPNetCMXDirection):
    def __init__(self, seed=23456, pretrained=False):
        super().__init__(seed=seed, pretrained=pretrained)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projector = ProjectionHead(64, 32)
        self.contrast_enabled = False
        self.embedding = None
        self.finalconv.register_forward_pre_hook(self._project)

    def _project(self, module, inputs):
        if self.training and self.contrast_enabled:
            self.embedding = self.projector(inputs[0])

    def forward(self, inputs):
        self.embedding = None
        return super().forward(inputs)
