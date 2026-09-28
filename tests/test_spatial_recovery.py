import numpy as np
from mechcal.generators.spatial_mixture import SpatialConfig,simulate_subject,observable_record
from recover_spatial_pilot import prepare,choice_probabilities


def test_fitting_likelihood_matches_generator():
    c=SpatialConfig();params=dict(gp_length=1.2,reward_temperature=.23,local_temperature=.7)
    for w,model in [(0.,'local'),(.4,'mixture'),(1.,'reward')]:
        e=simulate_subject(c,params,w,444);data=prepare(observable_record(c,'x',e),c)
        theta={'local':np.log([.7]),'reward':np.log([1.2,.23]),'mixture':np.r_[np.log([1.2,.23,.7]),w]}[model]
        p=choice_probabilities(theta,model,data,c)
        expected=np.take_along_axis(e['probabilities'],e['actions'][...,None],-1)[...,0]
        np.testing.assert_allclose(p,expected,atol=1e-12)
