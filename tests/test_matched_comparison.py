import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from run_matched_comparison import nll,metrics


def test_donor_loss_uses_last_action_axis():
    rng=np.random.default_rng(3)
    p=rng.uniform(.1,1,(3,2,200,4));p/=p.sum(-1,keepdims=True)
    a=rng.integers(0,4,(2,200))
    loss=nll(p,a[None])
    expected=np.array([[[-np.log(p[r,i,t,a[i,t]]) for t in range(200)] for i in range(2)] for r in range(3)])
    assert np.allclose(loss,expected)
    assert np.allclose(nll(p[:,:,199:200],a[None,:,199:200]),expected[:,:,199:200])


def test_matched_metrics_identity_and_scoring_window():
    rng=np.random.default_rng(4)
    p=rng.uniform(.1,1,(2,200,4));p/=p.sum(-1,keepdims=True)
    a=rng.integers(0,4,(2,200))
    donor=np.stack([p,p]);local=donor[:,:,199:200]
    v=metrics(p,donor,local,a,p,donor,local)
    assert np.allclose(v['intact_nll'],nll(p,a)[:,150:].mean(1))
    for k in ('donor_delta_nll','local200_delta_nll','donor_response_error','local200_response_error'):
        assert np.allclose(v[k],0)
