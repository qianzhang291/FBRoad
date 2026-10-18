"""HRNet-only selector; unused ResNet/custom-CUDA branches are not bundled."""
from .hrnet.hrnet_backbone import HRNetBackbone
class BackboneSelector:
    def __init__(self, configer):
        self.configer = configer
    def get_backbone(self):
        if self.configer.get('network', 'backbone') != 'hrnet2x20':
            raise ValueError('This release supports the experiment HRNet2x-W20 only')
        return HRNetBackbone(self.configer)()
