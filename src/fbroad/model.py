"""Same complete three-part model; training-only access to raw prediction scores."""
from .control_model import ContrastRoadNet as ExistingRoadNet


class ContrastRoadNet(ExistingRoadNet):
    def __init__(self, seed=23456, pretrained=False):
        super().__init__(seed=seed, pretrained=pretrained)
        self.raw_logits = None
        self.finalconv.register_forward_hook(self._capture_logits)

    def _capture_logits(self, module, inputs, output):
        if self.training:
            self.raw_logits = output

    def forward(self, inputs):
        self.raw_logits = None
        return super().forward(inputs)
