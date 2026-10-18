from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from fbroad.train import train_boundary
p=argparse.ArgumentParser(description='Train the independent boundary/direction corrector')
p.add_argument('--data-root',required=True)
p.add_argument('--split',default='configs/split.json')
p.add_argument('--seed',type=int,default=23456)
p.add_argument('--imagenet',required=True,help='Original HRNet2x-W20 ImageNet checkpoint')
p.add_argument('--validation-reference',required=True,help='Frozen same-seed lovasz/raw validation masks')
p.add_argument('--target-root',required=True)
p.add_argument('--output',required=True)
p.add_argument('--device',default='cuda')
p.add_argument('--workers',type=int,default=4)
p.add_argument('--resume',action='store_true')
train_boundary(p.parse_args())
