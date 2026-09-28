import numpy as np
from scipy.optimize._numdiff import approx_derivative
from run_rw_baseline import rw


def test_gradient_and_causality():
    rng=np.random.default_rng(4);a=rng.integers(4,size=(5,12));r=rng.normal(50,15,a.shape)
    for cue in (None,(np.array([0,1,2,3,0]),np.array([30.,40.,70.,60.,50.]))):
        theta=np.array([.32,np.log(5.)])
        f,g=rw(theta,a,r,4,cue,True)
        numeric=approx_derivative(lambda x:rw(x,a,r,4,cue,True)[0],theta).ravel()
        np.testing.assert_allclose(g,numeric,rtol=1e-5,atol=1e-7)
        p=rw(theta,a,r,4,cue);changed=r.copy();changed[:,5:]+=20
        q=rw(theta,a,changed,4,cue)
        np.testing.assert_array_equal(p[:,:6],q[:,:6])
        np.testing.assert_allclose(p.sum(-1),1)
        np.testing.assert_allclose(f,-np.log(np.take_along_axis(p,a[...,None],-1)).mean())


def test_zero_learning_rate_is_uniform():
    a=np.zeros((3,10),int);r=np.ones((3,10))*80
    np.testing.assert_allclose(rw([0,np.log(10)],a,r,4),.25)
