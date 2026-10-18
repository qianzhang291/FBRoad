from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import cv2
import numpy as np
from fbroad.data import load_split
from fbroad.metrics import confusion,scores,cldice
from fbroad.runtime import save_json
p=argparse.ArgumentParser(description='Evaluate predicted road masks on validation or test split')
p.add_argument('--predictions',required=True)
p.add_argument('--data-root',required=True)
p.add_argument('--split',default='configs/split.json')
p.add_argument('--part',choices=('validation','test'),required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
if Path(a.output).exists(): raise FileExistsError(a.output)
split=load_split(a.split)
root=Path(a.data_root)/('test' if a.part=='test' else 'train_val')/'mask'
rows=[];total=dict.fromkeys(('tp','fp','fn','tn'),0)
for sample in split[a.part]:
 pred=cv2.imread(str(Path(a.predictions)/f'{sample}.png'),0)
 gt=cv2.imread(str(root/f'{sample}_mask.png'),0)
 if pred is None or gt is None: raise FileNotFoundError(sample)
 if not set(np.unique(pred)).issubset({0,255}): raise ValueError('Expected binary 0/255 masks')
 c=confusion(pred>127,gt>127)
 for k in total: total[k]+=c[k]
 rows.append({'id':sample,**c,**scores(c),'cldice':cldice(pred>127,gt>127)})
save_json(a.output,{'count':len(rows),'confusion':total,'global_area_scores':scores(total),
                  'mean_image_cldice':float(np.mean([r['cldice'] for r in rows])),'per_image':rows})
