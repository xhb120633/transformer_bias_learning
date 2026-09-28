import numpy as np
from mechcal.generators.spatial_mixture import (SpatialConfig,GPLearner,coordinates,rbf,
    policy,replay,simulate_subject,observable_record)


PARAMS=dict(gp_length=1.2,reward_temperature=.2,local_temperature=.7)


def test_gp_online_matches_batch_posterior_with_repeats():
    x=coordinates(5); gp=GPLearner(x,1.2,.02)
    arms=[1,7,1,24];y=np.array([.2,-.7,.4,1.1]);k=rbf(x,1.2)
    for arm,reward in zip(arms,y): gp.observe(arm,reward)
    matrix=k[np.ix_(arms,arms)]+.02*np.eye(4)
    np.testing.assert_allclose(gp.mean,k[:,arms]@np.linalg.solve(matrix,y),atol=1e-12)
    np.testing.assert_allclose(gp.cov,k-k[:,arms]@np.linalg.solve(matrix,k[arms]),atol=1e-12)


def test_mixture_endpoints_and_affinity():
    x=coordinates(5);m=np.arange(25)/25
    f=lambda w:policy(m,x,12,w,.2,.7,.1)
    np.testing.assert_allclose(f(.3),.3*f(1)+.7*f(0))
    np.testing.assert_allclose(f(0),policy(-m,x,12,0,.2,.7,.1))
    assert not np.allclose(f(1),policy(-m,x,12,1,.2,.7,.1))
    assert np.all(f(.3)>=.1/25) and np.isclose(f(.3).sum(),1)


def test_replay_causal_and_zero_weight_reward_invariance():
    c=SpatialConfig(rounds=1,trials=8)
    e=simulate_subject(c,PARAMS,.5,123)
    for w in (0,.5,1):
        a=e['actions'][0];r=e['rewards'][0]
        p=replay(c,PARAMS,w,int(e['cue_arm'][0]),e['cue_reward'][0],a,r)
        altered=r.copy();altered[4:]+=40
        q=replay(c,PARAMS,w,int(e['cue_arm'][0]),e['cue_reward'][0],a,altered)
        np.testing.assert_array_equal(p[:5],q[:5])
        if w==0: np.testing.assert_array_equal(p,q)
        if w==.5: np.testing.assert_allclose(p,e['probabilities'][0],atol=1e-14)


def test_paired_environments_determinism_and_observable_separation():
    c=SpatialConfig(rounds=2,trials=5)
    e=simulate_subject(c,PARAMS,0,123);f=simulate_subject(c,PARAMS,1,123)
    for field in ('landscape','cue_arm','cue_reward','environment_seed'):
        np.testing.assert_array_equal(e[field],f[field])
    repeat=simulate_subject(c,PARAMS,0,123)
    np.testing.assert_array_equal(e['actions'],repeat['actions'])
    record=observable_record(c,'train_0000',e)
    assert set(record)=={'participant','grid_size','rounds'}
    assert set(record['rounds'][0])=={'round','initial_position','initial_reward','choices','rewards'}
    assert not np.array_equal(e['landscape'],simulate_subject(c,PARAMS,0,124)['landscape'])
