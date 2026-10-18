from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from fbroad.train import train_region
from fbroad.factory import ARMS
p=argparse.ArgumentParser(description='Train one region model; train/validation only')
p.add_argument('--data-root',required=True)
p.add_argument('--split',default='configs/split.json')
p.add_argument('--arm',choices=ARMS,default='ema')
p.add_argument('--seed',type=int,default=23456)
p.add_argument('--initial',required=True)
p.add_argument('--output',required=True)
p.add_argument('--device',default='cuda')
p.add_argument('--workers',type=int,default=4)
p.add_argument('--resume',action='store_true')
train_region(p.parse_args())
