from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from fbroad.train import prepare_initial
p=argparse.ArgumentParser(description='Create a seeded initial region model checkpoint')
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=23456)
p.add_argument('--imagenet',action='store_true',help='Load torchvision ResNet34 IMAGENET1K_V1 backbone weights')
a=p.parse_args()
prepare_initial(a.output,a.seed,a.imagenet)
