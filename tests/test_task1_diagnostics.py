import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from complete_task1_diagnostics import temper,calibrate,response_metrics

def test_temperature_identity_normalization():
    p=np.array([[[.1,.2,.3,.4]]])
    assert np.allclose(temper(p,1),p)
    assert np.allclose(temper(p,2).sum(-1),1)
    assert np.array_equal(temper(p,.3).argmax(-1),p.argmax(-1))

def test_crossfit_never_uses_held_fold_labels():
    rng=np.random.default_rng(3)
    p=rng.dirichlet(np.ones(4),size=(10,8));a=rng.integers(0,4,(10,8));folds=np.arange(10)%5
    _,t=calibrate(p,a,folds)
    altered=a.copy();altered[folds==0]=(altered[folds==0]+1)%4
    _,tt=calibrate(p,altered,folds)
    assert t[0]==tt[0]

def test_probability_response_direction_and_gain():
    p=np.array([[[.925,.025,.025,.025]]])
    q=np.array([[[[.025,.925,.025,.025]]]])
    v=response_metrics(p,q,p,q)
    assert np.allclose(v['response_error'],0)
    assert np.allclose(v['projection_gain'],1)
    assert np.allclose(v['response_cosine'],1)
    shrunk=p[None]+.2*(q-p[None])
    v=response_metrics(p,shrunk,p,q)
    assert np.allclose(v['projection_gain'],.2)
    assert np.allclose(v['response_cosine'],1)
    v=response_metrics(p,p[None],p,p[None])
    assert np.isnan(v['projection_gain']).all()
