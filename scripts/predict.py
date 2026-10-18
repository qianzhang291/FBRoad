from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import cv2
from fbroad.inference import RoadExtractor
from fbroad.runtime import seed_all
p=argparse.ArgumentParser(description='Extract a road mask from aligned image/GPS heatmap')
p.add_argument('--image',required=True)
p.add_argument('--gps',required=True)
p.add_argument('--region-checkpoint',required=True)
p.add_argument('--boundary-checkpoint')
p.add_argument('--arm',default='ema')
p.add_argument('--seed',type=int,default=23456)
p.add_argument('--device',default='cpu')
p.add_argument('--output',required=True)
a=p.parse_args()
seed_all(a.seed)
image,gps=cv2.imread(a.image),cv2.imread(a.gps,0)
if image is None or gps is None: raise FileNotFoundError('Image/GPS input missing')
model=RoadExtractor(a.region_checkpoint,a.boundary_checkpoint,a.arm,a.seed,a.device)
result=model.predict(image,gps)
path=Path(a.output)
if path.exists(): raise FileExistsError(path)
path.parent.mkdir(parents=True,exist_ok=True)
if not cv2.imwrite(str(path),result.astype('uint8')*255): raise IOError(path)
