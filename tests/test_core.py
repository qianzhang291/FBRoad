"""Unit tests for evaluation metrics, region network, EMA, and boundary refinement."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from fbroad.runtime import seed_all, save_checkpoint, load_checkpoint, rng_state, restore_rng
from fbroad.factory import make_region_model
from fbroad.metrics import confusion, scores, cldice
from fbroad.data import load_split

ROOT=Path(__file__).resolve().parents[1]

class MetricsAndRecipe(unittest.TestCase):
    def test_global_counts_are_not_image_mean(self):
        a=confusion(np.array([1,0,0,0],bool),np.array([1,0,0,0],bool))
        b=confusion(np.zeros(4,bool),np.ones(4,bool))
        c={k:a[k]+b[k] for k in a}
        self.assertEqual(scores(c)['iou'],.2)
    def test_cldice_empty_and_identical(self):
        z=np.zeros((9,9),bool);line=z.copy();line[4,:]=1
        self.assertEqual(cldice(z,z),1.)
        self.assertEqual(cldice(z,line),0.)
        self.assertEqual(cldice(line,line),1.)
    def test_original_split_is_mother_disjoint(self):
        s=load_split(ROOT/'configs/split.json')
        self.assertEqual([len(s[k]) for k in ('train','validation','test')],[1701,35,70])
    def test_checkpoint_rng_resume(self):
        seed_all(23456)
        state=rng_state()
        expected=torch.rand(5)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt'
            save_checkpoint(path,{'rng':state,'model':{'x':torch.ones(2)}})
            value=load_checkpoint(path)
        restore_rng(value['rng'])
        self.assertTrue(torch.equal(torch.rand(5),expected))
    def test_fixed_two_pixel_offset_axes(self):
        from fbroad.boundary import refine
        original=np.zeros((9,9),bool);original[:,4]=1
        b=torch.zeros(1,2,9,9);b[:,1]=10
        d=torch.full((1,8,9,9),-10.);d[:,4]=10 # (dy,dx)=(0,+1)
        corrected,_=refine(original,b,d)
        expected=np.zeros_like(original);expected[:,2]=1
        self.assertTrue(np.array_equal(corrected,expected))
    def test_boundary_forward_and_backward(self):
        from fbroad.boundary import make_model
        from fbroad.segfix_loss import segfix_loss
        seed_all(23456)
        model=make_model().train()
        b,d=model(torch.randn(2,3,64,64))
        target=torch.randint(0,2,(2,64,64))
        loss,_=segfix_loss(b,d,target,torch.randint(0,8,target.shape))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.mask_head[0].weight.grad).all())
        self.assertTrue(torch.isfinite(model.dir_head[0].weight.grad).all())

class RegionNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(23456)
        cls.model=make_region_model('ema').eval()
    def test_original_state_schema_and_forward(self):
        self.assertEqual(len(self.model.state_dict()),905)
        with torch.no_grad():
            y=self.model(torch.randn(2,4,64,64))
        self.assertEqual(tuple(y.shape),(2,1,64,64))
        self.assertTrue(torch.isfinite(y).all())
    def test_disabled_additions_restore_baseline(self):
        m=self.model
        baseline=make_region_model('baseline').eval()
        baseline.load_state_dict({k:m.state_dict()[k] for k in baseline.state_dict()},strict=True)
        m.fusion_enabled=False;m.direction_enabled=False
        x=torch.randn(2,4,64,64)
        with torch.no_grad():
            actual,expected=m(x),baseline(x)
        m.fusion_enabled=True;m.direction_enabled=True
        self.assertTrue(torch.equal(actual,expected))
    def test_late_loss_and_input_gradients(self):
        from fbroad.contrast import PixelContrastLoss
        from fbroad.iou_objective import road_objective
        from fbroad._vendor.cmmpnet.loss import dice_bce_loss
        m=self.model.train();m.contrast_enabled=True;m.zero_grad(set_to_none=True)
        x=torch.randn(2,4,64,64,requires_grad=True)
        p=m(x);target=(torch.rand(2,1,128,128)>.75).float()
        main,_=road_objective(dice_bce_loss(),target,p,m.raw_logits,True)
        aux,_=PixelContrastLoss()(m.embedding,target,p,seed=23457)
        (main+.01*aux).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:,:3].abs().sum()),0.)
        self.assertGreater(float(x.grad[:,3:].abs().sum()),0.)
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all()
                            for t in m.parameters() if t.requires_grad))
        m.embedding=None;m.raw_logits=None;m.contrast_enabled=False;m.eval()
    def test_ema_selected_weights_survive_save(self):
        from fbroad.reference_ema import ModelEmaV3
        m=torch.nn.Linear(2,1)
        shadow=ModelEmaV3(m,decay=.99,update_after_step=0,use_warmup=False,
                         foreach=True,exclude_buffers=False)
        with torch.no_grad(): m.weight.add_(1)
        shadow.update(m,step=1)
        with torch.no_grad(): m.weight.add_(1)
        shadow.update(m,step=2)
        expected=shadow.module.weight.detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'best.pt'
            save_checkpoint(path,{'model':shadow.module.state_dict(),'raw_model':m.state_dict(),
                                  'ema':shadow.state_dict()})
            value=load_checkpoint(path)
        self.assertTrue(torch.equal(value['model']['weight'],expected))
        self.assertFalse(torch.equal(value['model']['weight'],value['raw_model']['weight']))
        other=ModelEmaV3(torch.nn.Linear(2,1),decay=.99,update_after_step=0,
                         use_warmup=False,foreach=True,exclude_buffers=False)
        other.load_state_dict(value['ema'],strict=True)
        self.assertTrue(torch.equal(other.module.weight,expected))

if __name__=='__main__':
    unittest.main()
