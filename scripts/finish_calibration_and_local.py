"""Consume validation-only temperature exports or independently local rewards."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from mechcal.analysis.restless_recovery import load_split
from mechcal.analysis.generator_oracle_donor import _ordered_condition

ROOT=Path(__file__).resolve().parents[1]
CS=('reward_dominant','balanced','choice_dominant')


def scale_probs(p,temp):
    lp=np.log(np.clip(p,1e-30,1))/temp
    return np.exp(lp-logsumexp(lp,axis=-1,keepdims=True))


def nll(p,actions):
    return -np.log(np.take_along_axis(p,actions[...,None],-1).squeeze(-1))


def interval(v):
    rng=np.random.default_rng(20260906)
    means=v[rng.integers(len(v),size=(10000,len(v)))].mean(-1)
    return {'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}


def temperature():
    data=load_split(ROOT/'outputs/restless_pooled_eps10_1000_20260827','test')
    result={}
    for c in CS:
        with np.load(ROOT/f'outputs/centaur_calibration_val/{c}/trial_losses.npz') as z:
            vp=z['intact_choice_probabilities'].astype(float); vi=z['participants'].astype(str)
        records=[json.loads(l) for l in (ROOT/f'outputs/centaur_sft_eps10_seed11/{c}_val.jsonl').read_text().splitlines()]
        lookup={str(r['participant']):['ABCD'.index(a) for a in re.findall(r'<<([ABCD])>>',r['text'])] for r in records}
        va=np.asarray([lookup[i] for i in vi]); assert va.shape==vp.shape[:-1]
        fit=minimize_scalar(lambda x:nll(scale_probs(vp,np.exp(x)),va).mean(),bounds=(np.log(.05),np.log(20)),method='bounded')
        if not fit.success: raise RuntimeError(fit.message)
        temp=float(np.exp(fit.x))
        with np.load(ROOT/f'outputs/centaur_donor_probabilities_test/{c}/trial_losses.npz') as z:
            ids=z['participants'].astype(str); ip=z['intact_choice_probabilities'].astype(float); dp=z['donor_choice_probabilities'].astype(float)
        assert not set(ids)&set(vi)
        d=_ordered_condition(data,c,ids); a=d['action'].astype(int)
        result[c]={'temperature':temp,'validation_nll_before':float(nll(scale_probs(vp,1),va).mean()),'validation_nll_after':float(fit.fun),'test':{}}
        for name,t in [('original',1.),('validation_calibrated',temp)]:
            p=scale_probs(ip,t); q=scale_probs(dp,t)
            il=nll(p,a); dl=nll(q,a[None])
            result[c]['test'][name]={}
            for s,m in [('all',np.ones(len(ids),bool)),('sign_+1',d['kernel_sign']==1),('sign_-1',d['kernel_sign']==-1)]:
                result[c]['test'][name][s]={'intact_nll':interval(il[m].mean(1)),'delta_nll':interval((dl[:,m]-il[None,m]).mean((0,2))),'tv':interval((.5*abs(q[:,m]-p[None,m]).sum(-1)).mean((0,2)))}
    return {'method':'temperature selected only using intact validation choice NLL; ABCD normalized; paired participant bootstrap','conditions':result}


def oracle_local(d,mappings,trials,scale):
    n=len(d['action']); rows=np.arange(n); q=np.full((n,4),50.)
    def policy(state,t):
        k=np.zeros((n,4))
        if t: k[rows,d['action'][:,t-1]]=1
        logits=d['beta_reward'][None,:,None]*(state-state.mean(-1,keepdims=True))/scale+d['beta_kernel'][None,:,None]*k[None]
        w=np.isclose(logits,logits.max(-1,keepdims=True),atol=1e-12,rtol=0)
        return .025+.9*w/w.sum(-1,keepdims=True)
    intact=[]; donor=[]
    for t in range(200):
        if t+1 in trials:
            intact.append(policy(q[None],t)[0])
            states=np.repeat(q[None],len(mappings),axis=0)
            if t:
                prev=d['action'][:,t-1]
                change=d['alpha'][None]*(d['reward'][mappings,t-1]-d['reward'][None,:,t-1])
                states[np.arange(len(mappings))[:,None],rows[None],prev[None]]+=change
            donor.append(policy(states,t))
        a=d['action'][:,t]; q[rows,a]+=d['alpha']*(d['reward'][:,t]-q[rows,a])
    return np.stack(intact,1),np.stack(donor,2)


def local():
    ds=ROOT/'outputs/restless_pooled_eps10_1000_20260827'
    data=load_split(ds,'test'); scale=json.loads((ds/'metadata.json').read_text())['q_pairwise_scale']
    result={}
    for c in CS:
        with np.load(ROOT/f'outputs/centaur_local_reward_test/{c}/trial_losses.npz') as z:
            ids=z['participants'].astype(str); mappings=z['donor_mappings']; trials=z['trial_indices']
            mi=z['intact_choice_probabilities'].astype(float); md=z['donor_choice_probabilities'].astype(float)
        assert list(trials)==[1,2,10,50,100,200]
        d=_ordered_condition(data,c,ids); oi,od=oracle_local(d,mappings,trials,scale)
        with np.load(ROOT/f'outputs/centaur_donor_probabilities_test/{c}/trial_losses.npz') as full:
            assert np.array_equal(ids,full['participants'].astype(str))
            assert np.array_equal(mappings,full['donor_mappings'])
            prefix_error=float(np.max(abs(mi-full['intact_choice_probabilities'][:,trials-1])))
        assert np.isfinite(mi).all() and np.isfinite(md).all()
        mass_min=float(min(mi.sum(-1).min(),md.sum(-1).min()))
        mi=scale_probs(mi,1); md=scale_probs(md,1)
        assert np.max(abs(md[:,:,0]-mi[None,:,0]))<1e-6
        assert np.max(abs(od[:,:,0]-oi[None,:,0]))<1e-12
        actions=d['action'][:,trials-1].astype(int)
        model_delta=nll(md,actions[None])-nll(mi,actions)[None]
        oracle_delta=nll(od,actions[None])-nll(oi,actions)[None]
        result[c]={'audit':{'prefix_vs_full_probability_max_abs':prefix_error,'valid_choice_mass_min':mass_min,'trial1_null_passed':True}}
        for s,m in [('all',np.ones(len(ids),bool)),('sign_+1',d['kernel_sign']==1),('sign_-1',d['kernel_sign']==-1)]:
            result[c][s]={}
            for ti,t in enumerate(trials):
                dm=md[:,m,ti]-mi[None,m,ti]; do=od[:,m,ti]-oi[None,m,ti]
                mtv=(.5*abs(dm).sum(-1)).mean(0); otv=(.5*abs(do).sum(-1)).mean(0)
                ml=model_delta[:,m,ti].mean(0); ol=oracle_delta[:,m,ti].mean(0)
                result[c][s][str(t)]={'model_tv':interval(mtv),'oracle_tv':interval(otv),'oracle_minus_model_tv':interval(otv-mtv),'response_error_tv':interval((.5*abs(dm-do).sum(-1)).mean(0)),'model_delta_nll':interval(ml),'oracle_delta_nll':interval(ol),'oracle_minus_model_delta_nll':interval(ol-ml)}
    return {'method':'independent preceding reward replacement; six fixed trials, not an all-trials average','conditions':result}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['temperature','local']); args=parser.parse_args()
    result=temperature() if args.mode=='temperature' else local()
    out=ROOT/f'outputs/{args.mode}_comparison_20260906'; out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
