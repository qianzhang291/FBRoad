from ._vendor.cmmpnet.networks.CMMPNet import DinkNet34_CMMPNet
from .core_model import CMMPNetCMX
from .backbone import CMMPNetCMXDirection
from .control_model import ContrastRoadNet as ContrastModel
from .model import ContrastRoadNet

ARMS = ('baseline', 'fusion', 'direction', 'contrast', 'lovasz', 'ema')

def make_region_model(arm='ema', seed=23456, pretrained=False):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm {arm!r}; expected {ARMS}')
    if arm == 'baseline':
        return DinkNet34_CMMPNet(pretrained=pretrained)
    cls = {'fusion': CMMPNetCMX, 'direction': CMMPNetCMXDirection,
           'contrast': ContrastModel, 'lovasz': ContrastRoadNet, 'ema': ContrastRoadNet}[arm]
    return cls(seed=seed, pretrained=pretrained)
