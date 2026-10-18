from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import cv2
from fbroad.inference import RoadExtractor
from fbroad.data import load_split
from fbroad.runtime import seed_all
p=argparse.ArgumentParser(description='Run inference over a dataset split (train, validation, or test)')
p.add_argument('--data-root',required=True)
p.add_argument('--split',default='configs/split.json')
p.add_argument('--part',choices=('train','validation','test'),required=True)
p.add_argument('--region-checkpoint',required=True)
p.add_argument('--boundary-checkpoint')
p.add_argument('--arm',default='ema')
p.add_argument('--seed',type=int,default=23456)
p.add_argument('--device',default='cpu')
p.add_argument('--output',required=True)
a=p.parse_args(); seed_all(a.seed)
split=load_split(a.split)
out=Path(a.output)
if out.exists() and any(out.iterdir()): raise FileExistsError(out)
out.mkdir(parents=True,exist_ok=True)
model=RoadExtractor(a.region_checkpoint,a.boundary_checkpoint,a.arm,a.seed,a.device)
root=Path(a.data_root)/('test' if a.part=='test' else 'train_val')
for sample in split[a.part]:
 image=cv2.imread(str(root/'image'/f'{sample}_sat.png'))
 gps=cv2.imread(str(root/'gps'/f'{sample}_gps.jpg'),0)
 if image is None or gps is None: raise FileNotFoundError(sample)
 result=model.predict(image,gps)
 if not cv2.imwrite(str(out/f'{sample}.png'),result.astype('uint8')*255): raise IOError(sample)
