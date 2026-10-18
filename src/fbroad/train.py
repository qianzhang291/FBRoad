"""Training and initialization routines for region and boundary networks."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_split, region_loader
from .factory import make_region_model
from .runtime import seed_all, save_checkpoint, save_json, rng_state, restore_rng, load_checkpoint
from .metrics import confusion, scores, restore_probability
from ._vendor.cmmpnet.loss import dice_bce_loss
from .contrast import PixelContrastLoss
from .iou_objective import road_objective
from .reference_ema import ModelEmaV3


def _new_output(path, resume):
    path = Path(path)
    if resume:
        if not (path/'last.pt').is_file():
            raise ValueError('Resume requires an existing last.pt')
    elif path.exists() and any(path.iterdir()):
        raise FileExistsError('Use a new output directory or explicitly resume')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value) + '\n')


@torch.no_grad()
def validate_region(model, loader, device):
    model.eval()
    totals = dict.fromkeys(('tp','fp','fn','tn'), 0)
    for x, y, _ in loader:
        values = model(x.to(device)).cpu().numpy()
        for p, t in zip(values, y.numpy()):
            c = confusion(restore_probability(p[0], t[0].shape) > .5, t[0] > .5)
            for k in totals:
                totals[k] += c[k]
    return {**scores(totals), **totals}


def prepare_initial(output, seed, imagenet=False):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    seed_all(seed)
    model = make_region_model('ema', seed, pretrained=imagenet)
    save_checkpoint(path, {'model':model.state_dict(), 'rng':rng_state(), 'seed':seed,
                           'imagenet_backbones':imagenet})


def train_region(args):
    seed_all(args.seed)
    split = load_split(args.split)
    out = _new_output(args.output, args.resume)
    device = torch.device(args.device)
    model = make_region_model(args.arm, args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    shadow = None
    if args.arm == 'ema':
        shadow = ModelEmaV3(model, decay=.99, update_after_step=0,
                            use_warmup=False, foreach=True, exclude_buffers=False)
        shadow.module.requires_grad_(False).eval()
    state = {'epoch':0, 'steps':0, 'samples':0, 'best_iou':-1., 'best_epoch':0}
    if args.resume:
        value = load_checkpoint(out/'last.pt')
        if value['arm'] != args.arm or value['seed'] != args.seed:
            raise ValueError('Resume seed/arm differs')
        model.load_state_dict(value.get('raw_model',value['model']), strict=True)
        optimizer.load_state_dict(value['optimizer'])
        state = value['state']
        if shadow is not None:
            shadow.load_state_dict(value['ema'], strict=True)
        restore_rng(value['rng'])
    else:
        value = load_checkpoint(args.initial)
        if value['seed'] != args.seed:
            raise ValueError('Initial seed differs')
        model.load_state_dict({k:value['model'][k] for k in model.state_dict()}, strict=True)
        if shadow is not None:
            shadow.module.load_state_dict(model.state_dict(), strict=True)
        restore_rng(value['rng'])
    criterion, contrastive = dice_bce_loss(), PixelContrastLoss()
    save_json(out/'recipe.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    for epoch in range(state['epoch']+1, 31):
        model.train()
        late = epoch >= 16
        if hasattr(model,'contrast_enabled'):
            model.contrast_enabled = late and args.arm in ('contrast','lovasz','ema')
        lr = 1e-4 if epoch <= 16 else 1e-5
        for g in optimizer.param_groups:
            g['lr'] = lr
        losses = []
        for x,y,ids in region_loader(split['train'],args.data_root,'train_val',args.seed,
                                     8,epoch,True,args.workers):
            if (out/'STOP').exists():
                raise InterruptedError('STOP: last complete epoch checkpoint is retained')
            optimizer.zero_grad(set_to_none=True)
            y = y.to(device)
            p = model(x.to(device))
            main, detail = road_objective(criterion,y,p,getattr(model,'raw_logits',None),
                                         late and args.arm in ('lovasz','ema'))
            auxiliary = main.new_zeros(())
            if getattr(model,'contrast_enabled',False):
                auxiliary,_ = contrastive(model.embedding,y,p,seed=args.seed+state['steps']+1)
            loss = main + .01*auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite region loss')
            loss.backward()
            if hasattr(model,'embedding'): model.embedding = None
            if hasattr(model,'raw_logits'): model.raw_logits = None
            optimizer.step()
            state['steps'] += 1
            state['samples'] += len(ids)
            if shadow is not None:
                shadow.update(model,step=state['steps'])
            losses.append(float(loss.detach()))
        selected = shadow.module if shadow is not None else model
        val = validate_region(selected,region_loader(split['validation'],args.data_root,
                              'train_val',args.seed,8,workers=args.workers),device)
        improved = val['iou'] > state['best_iou']
        state['epoch'] = epoch
        if improved:
            state.update(best_iou=val['iou'],best_epoch=epoch)
        payload = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'rng':rng_state(),'state':dict(state),'arm':args.arm,'seed':args.seed}
        if shadow is not None:
            payload.update(raw_model=model.state_dict(),ema=shadow.state_dict())
        save_checkpoint(out/'last.pt',payload)
        if improved:
            payload['model'] = selected.state_dict()
            payload['model_field_is_selected_ema'] = shadow is not None
            save_checkpoint(out/'best.pt',payload)
        _append(out/'epochs.jsonl',{'epoch':epoch,'state':dict(state),'validation':val,
                                   'mean_loss':float(np.mean(losses)),'selected':improved})
    save_json(out/'complete.json',state)


def prepare_boundary_targets(split, data_root, target_root):
    from .boundary_data import targets_from_mask
    target_root = Path(target_root)
    target_root.mkdir(parents=True,exist_ok=True)
    for sample in split['train']:
        dst = target_root/f'{sample}.npy'
        if dst.exists():
            if np.load(dst,mmap_mode='r').shape != (1024,1024,2):
                raise ValueError(f'Invalid target: {dst}')
            continue
        mask = cv2.imread(str(Path(data_root)/'train_val/mask'/f'{sample}_mask.png'),0)
        if mask is None: raise FileNotFoundError(sample)
        tmp = dst.with_suffix('.tmp.npy')
        np.save(tmp,targets_from_mask(mask>127))
        tmp.replace(dst)


@torch.no_grad()
def validate_boundary(model, split, data_root, reference, device):
    from .boundary import normalize_bgr, refine
    model.eval()
    totals = dict.fromkeys(('tp','fp','fn','tn'),0)
    for sample in split['validation']:
        root = Path(data_root)/'train_val'
        image = cv2.imread(str(root/'image'/f'{sample}_sat.png'))
        truth = cv2.imread(str(root/'mask'/f'{sample}_mask.png'),0)
        original = cv2.imread(str(Path(reference)/f'{sample}.png'),0)
        if image is None or truth is None or original is None:
            raise FileNotFoundError(f'Validation inputs/reference missing: {sample}')
        b,d = model(normalize_bgr(image).unsqueeze(0).to(device))
        corrected,_ = refine(original>127,b,d)
        c = confusion(corrected,truth>127)
        for k in totals: totals[k] += c[k]
    return {**scores(totals),**totals}


def train_boundary(args):
    from .boundary import make_model
    from .boundary_data import TrainDataset
    from .segfix_loss import segfix_loss
    seed_all(args.seed)
    split = load_split(args.split)
    out = _new_output(args.output,args.resume)
    device = torch.device(args.device)
    # Reference masks MUST come from the same-seed frozen lovasz/raw model.
    for sample in split['validation']:
        if not (Path(args.validation_reference)/f'{sample}.png').is_file():
            raise FileNotFoundError(f'Missing frozen validation reference {sample}')
    prepare_boundary_targets(split,args.data_root,args.target_root)
    model = make_model(None if args.resume else args.imagenet).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005)
    state = {'epoch':0,'steps':0,'samples':0,'best_iou':-1.,'best_epoch':0}
    if args.resume:
        value = load_checkpoint(out/'last.pt')
        if value['seed'] != args.seed: raise ValueError('Resume seed differs')
        model.load_state_dict(value['model'],strict=True)
        optimizer.load_state_dict(value['optimizer'])
        restore_rng(value['rng'])
        state = value['state']
    save_json(out/'recipe.json',vars(args))
    max_steps = 60*((len(split['train'])+3)//4)
    for epoch in range(state['epoch']+1,61):
        loader = DataLoader(TrainDataset(split,epoch,args.data_root,args.target_root,args.seed),
            batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(args.seed+epoch),
            num_workers=args.workers,pin_memory=True,drop_last=False)
        model.train()
        for image,boundary,direction,ids in loader:
            if (out/'STOP').exists(): raise InterruptedError('STOP; epoch checkpoint retained')
            for g in optimizer.param_groups:
                g['lr'] = .01*(1-state['steps']/max_steps)**.9
            optimizer.zero_grad(set_to_none=True)
            b,d = model(image.to(device))
            loss,_ = segfix_loss(b,d,boundary.to(device),direction.to(device))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite boundary loss')
            loss.backward();optimizer.step()
            state['steps'] += 1;state['samples'] += len(ids)
        val = validate_boundary(model,split,args.data_root,args.validation_reference,device)
        improved = val['iou'] > state['best_iou']+1e-12
        state['epoch'] = epoch
        if improved: state.update(best_iou=val['iou'],best_epoch=epoch)
        payload = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'rng':rng_state(),'state':dict(state),'seed':args.seed}
        save_checkpoint(out/'last.pt',payload)
        if improved: save_checkpoint(out/'best.pt',payload)
        _append(out/'epochs.jsonl',{'state':dict(state),'validation':val,'selected':improved})
    save_json(out/'complete.json',state)
