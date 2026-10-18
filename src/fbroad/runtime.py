import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch

def seed_all(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def rng_state():
    value = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        value['cuda'] = torch.cuda.get_rng_state_all()
    return value

def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if 'cuda' in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value['cuda'])

def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def load_checkpoint(path):
    # Original experiment files also contain optimizer/RNG objects. Use trusted files.
    return torch.load(path, map_location='cpu', weights_only=False)
