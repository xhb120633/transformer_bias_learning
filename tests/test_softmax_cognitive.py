import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from fit_softmax_cognitive import objective,soft_replay,fit_one


def test_gradient_and_replay():
    rng=np.random.default_rng(7)
    a=rng.integers(0,4,40);r=rng.integers(0,101,40).astype(float)
    x=np.array([.27,.63,1.3])
    for sign in (-1.,1.):
        value,g=objective(x,a,r,14.8,sign)
        numeric=approx_derivative(lambda z: objective(z,a,r,14.8,sign)[0],x).ravel()
        assert np.allclose(g,numeric,rtol=1e-5,atol=1e-5)
        d=dict(action=a[None],reward=r[None])
        p,l=soft_replay(d,d['reward'],np.array([[*x,sign]]),14.8)
        assert np.allclose(p.sum(-1),1)
        assert np.isclose(l.sum(),value)
        changed=r.copy();changed[20:]=100-changed[20:]
        assert np.array_equal(p[:,:21],soft_replay(d,changed[None],np.array([[*x,sign]]),14.8)[0][:,:21])


def test_multiple_starts_finite():
    rng=np.random.default_rng(8)
    result=fit_one((rng.integers(0,4,40),rng.uniform(0,100,40),14.8))
    assert result.shape==(24,7)
    assert np.isfinite(result).all()


def test_local_replay_and_zero_weight():
    rng=np.random.default_rng(9)
    d=dict(action=rng.integers(0,4,(2,200)),reward=rng.uniform(0,100,(2,200)))
    pa=np.array([[.2,.4,1.,-1.],[.3,.7,2.,1.]])
    rewards=100-d['reward']
    p,_=soft_replay(d,rewards,pa,14.8,local=True)
    for t in (1,100,150,199):
        changed=d['reward'].copy();changed[:,t-1]=rewards[:,t-1]
        assert np.allclose(p[:,t],soft_replay(d,changed,pa,14.8)[0][:,t],atol=1e-12)
    pa[:,1]=0
    assert np.array_equal(soft_replay(d,rewards,pa,14.8)[0],soft_replay(d,d['reward'],pa,14.8)[0])
