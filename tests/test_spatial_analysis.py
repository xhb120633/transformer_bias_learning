import numpy as np
from ablate_spatial_references import nll


def test_nll_broadcast_matches_explicit_seed_donor_loops():
    rng=np.random.default_rng(11)
    p=rng.uniform(.1,1,(3,20,5,2,20,25));p/=p.sum(-1,keepdims=True)
    a=rng.integers(25,size=(5,2,20))
    vectorized=nll(p,a[None,None]).mean((1,3,4))
    expected=np.array([[np.mean([-np.log(p[s,d,i,b,t,a[i,b,t]])
        for d in range(20) for b in range(2) for t in range(20)])
        for i in range(5)] for s in range(3)])
    np.testing.assert_allclose(vectorized,expected)
