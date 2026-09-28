"""Independent checks of prefix fitting and exact epsilon-greedy replay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import torch
from run_reference_followup import grid_fit, params
from run_weight_curve_a import replay, nll


def toy():
    rng=np.random.default_rng(2)
    return dict(action=rng.integers(0,4,(2,200)),reward=rng.integers(0,101,(2,200)),
                alpha=np.array([.25,.35]),beta_reward=np.array([.5,.5]),
                beta_kernel=np.array([.5,-.5]),decision_noise=np.full(2,.1))


def test_prefix_causality():
    d=toy(); p=replay(d,d['reward'],15.)
    r=d['reward'].copy(); r[:,100:]=100-r[:,100:]
    assert np.array_equal(p[:,:101],replay(d,r,15.)[:,:101])
    a=d['action'].copy(); a[:,100:]=(a[:,100:]+1)%4
    assert np.array_equal(p[:,:101],replay(dict(d,action=a),d['reward'],15.)[:,:101])


def test_local_and_endpoints():
    d=toy(); r=100-d['reward']
    local=replay(d,r,15.,True)
    for t in (1,50,150,199):
        rr=d['reward'].copy(); rr[:,t-1]=r[:,t-1]
        assert np.allclose(local[:,t],replay(d,rr,15.)[:,t],atol=1e-12)
    zero=params(d,d['alpha'],np.zeros(2),np.array([1,-1]))
    assert np.array_equal(replay(zero,r,15.),replay(zero,d['reward'],15.))
    one=params(d,d['alpha'],np.ones(2),np.ones(2))
    minus=params(d,d['alpha'],np.ones(2),-np.ones(2))
    assert np.array_equal(replay(one,r,15.),replay(minus,r,15.))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='grid fitter uses CUDA')
def test_grid_returns_actual_candidates_with_reproducible_loss():
    d=toy()
    fit=grid_fit(d['action'][:,:150],d['reward'][:,:150],15.)
    for k in range(8):
        pa=fit['parameters'][:,k]
        f=params(d,pa[:,0],pa[:,1],pa[:,2])
        loss=nll(replay(f,d['reward'],15.),d['action'])[:,:150].sum(1)
        assert np.allclose(loss,fit['fit_total_nll'][:,k],atol=1e-7)
        assert np.all(loss<=fit['fit_total_nll'][:,0]+1+1e-7)
