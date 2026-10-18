import cv2
import numpy as np
from skimage.morphology import skeletonize

def confusion(prediction, truth):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if p.shape != t.shape:
        raise ValueError('Prediction and mask shapes differ')
    return {'tp': int((p&t).sum()), 'fp': int((p&~t).sum()),
            'fn': int((~p&t).sum()), 'tn': int((~p&~t).sum())}

def scores(c):
    tp, fp, fn = (c[k] for k in ('tp','fp','fn'))
    return {'iou': tp/max(tp+fp+fn,1), 'precision': tp/max(tp+fp,1),
            'recall': tp/max(tp+fn,1), 'f1': 2*tp/max(2*tp+fp+fn,1)}

def cldice(prediction, truth):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if not p.any() or not t.any():
        return float(not p.any() and not t.any())
    sp, st = skeletonize(p, method='zhang'), skeletonize(t, method='zhang')
    prec, sens = (sp&t).sum()/sp.sum(), (st&p).sum()/st.sum()
    return float(2*prec*sens/(prec+sens)) if prec+sens else 0.

def restore_probability(value, shape=(1024,1024)):
    return cv2.resize(value, (shape[1],shape[0]), interpolation=cv2.INTER_LINEAR)
