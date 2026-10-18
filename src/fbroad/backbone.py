"""One published four-direction decoder branch in the RGB late decoder."""
import torch
from torch import nn
from .core_model import CMMPNetCMX
from ._vendor.cmmpnet.networks.basic_blocks import DecoderBlock1DConv4


class CMMPNetCMXDirection(CMMPNetCMX):
    def __init__(self, seed=23456, pretrained=False):
        super().__init__(seed=seed, pretrained=pretrained)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.direction_decoder = DecoderBlock1DConv4(128, 64)
        self.direction_alpha = nn.Parameter(torch.zeros(()))
        self.direction_enabled = True
        self.decoder2.register_forward_hook(self._extend)

    def _extend(self, module, inputs, output):
        if not self.direction_enabled:
            return output
        extension = self.direction_decoder(inputs[0])
        assert extension.shape == output.shape
        return output + self.direction_alpha * extension
