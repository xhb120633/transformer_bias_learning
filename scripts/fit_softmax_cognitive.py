"""Pure-softmax RL + signed one-back kernel: prefix MLE and held-out assays."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp
from run_reference_followup import ROOT, WEIGHTS, load, groups
from run_weight_curve_a import replay, nll, derangements

BOUNDS=((.0001,.9999),(0.,1.),(-6.,6.))
STARTS=list(product((.1,.3,.7),(.2,.8),(0.,np.log(8.))))


@njit(cache=True, nogil=True)
def objective(x, actions, rewards, scale, sign):
    alpha,w,logbeta=x
    beta=np.exp(logbeta)
    q=np.full(4,50.)
    dq=np.zeros(4)
    loss=0.
    grad=np.zeros(3)
    for t in range(len(actions)):
        z=(q-q.mean())/scale
        dz=(dq-dq.mean())/scale
        k=np.zeros(4)
        if t: k[actions[t-1]]=1.
        v=w*z+(1-w)*sign*k
        logits=beta*v
        mx=logits.max()
        ex=np.exp(logits-mx)
        p=ex/ex.sum()
        a=actions[t]
        loss+=mx+np.log(ex.sum())-logits[a]
        p[a]-=1.
        grad[0]+=(p*beta*w*dz).sum()
        grad[1]+=(p*beta*(z-sign*k)).sum()
        grad[2]+=(p*logits).sum()
        dq[a]=(1-alpha)*dq[a]+rewards[t]-q[a]
        q[a]+=alpha*(rewards[t]-q[a])
    return loss,grad


def fit_one(payload):
    actions,rewards,scale=payload
    solutions=[]
    for sign in (-1.,1.):
        for start in STARTS:
            fit=minimize(objective,np.array(start),args=(actions,rewards,scale,sign),jac=True,
                         method='L-BFGS-B',bounds=BOUNDS,
                         options=dict(maxiter=500,ftol=1e-12,gtol=1e-7,maxls=50))
            solutions.append([*fit.x,sign,fit.fun,float(fit.success),fit.nit])
    solutions=np.asarray(solutions)
    assert np.isfinite(solutions).all()
    return solutions


def soft_replay(d, rewards, pa, scale, local=False):
    n,tmax=d['action'].shape
    q=np.full((n,4),50.)
    alpha,w,lb,sign=pa.T
    rows=np.arange(n)
    logp=np.empty((n,tmax,4))
    for t in range(tmax):
        changed=q.copy()
        if local and t:
            changed[rows,d['action'][:,t-1]]+=alpha*(rewards[:,t-1]-d['reward'][:,t-1])
        z=(changed-changed.mean(1,keepdims=True))/scale
        logits=w[:,None]*z
        if t: logits[rows,d['action'][:,t-1]]+=(1-w)*sign
        logits*=np.exp(lb[:,None])
        logp[:,t]=logits-logsumexp(logits,axis=1,keepdims=True)
        a=d['action'][:,t]
        r=d['reward'] if local else rewards
        q[rows,a]+=alpha*(r[:,t]-q[rows,a])
    return np.exp(logp),-np.take_along_axis(logp,d['action'][...,None],2)[...,0]


def assay(d,pa,scale,maps):
    p,base=soft_replay(d,d['reward'],pa,scale)
    op=replay(d,d['reward'],scale); ob=nll(op,d['action'])
    metrics=dict(intact_nll=base[:,150:].mean(1),oracle_intact_nll=ob[:,150:].mean(1),
                 intact_tv=(.5*np.abs(p-op).sum(-1))[:,150:].mean(1))
    arrays=dict(participants=d['base_participant_id'],actions=d['action'],parameters=pa,
                intact_probabilities=p,intact_nll=base,oracle_intact_nll=ob,donor_mappings=maps)
    for mode in ('donor','local','suffix_donor'):
        delta,odelta,error=[],[],[]
        for m in maps:
            r=d['reward'][m].copy()
            if mode=='suffix_donor': r[:,:150]=d['reward'][:,:150]
            pp,loss=soft_replay(d,r,pa,scale,mode=='local')
            oo=replay(d,r,scale,mode=='local')
            delta.append(loss-base); odelta.append(nll(oo,d['action'])-ob)
            error.append(.5*np.abs((pp-p)-(oo-op)).sum(-1))
            assert np.allclose(pp[:,0],p[:,0],atol=1e-12)
        for key,values in ((mode+'_delta_nll',delta),('oracle_'+mode+'_delta_nll',odelta),(mode+'_response_error',error)):
            v=np.stack(values);arrays[key]=v;metrics[key]=v[:,:,150:].mean((0,2))
    return groups(metrics,d['kernel_sign']),arrays


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scale=json.loads((ROOT/'protocol.json').read_text())['q_scale']
    protocol=dict(model='pure softmax RL plus signed one-back kernel; no lapse',fit_trials='1-150',
        evaluation_trials='151-200',split='val',parameters=['alpha','reward_weight','log_inverse_temperature','kernel_sign'],
        starts_per_sign=12,bounds=BOUNDS,starts=STARTS,limit=args.limit,
        selection='lowest finite prefix NLL among all 24 terminal solutions; success flags retained',
        known='initial Q=50 and fixed Q scale; no true individual parameters or condition bounds',
        caveat='Generator uses epsilon greedy; fitted weights are not guaranteed to recover its numerical weights.',
        test_used=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    objective(np.array([.3,.5,1.]),np.array([0,1]),np.array([50.,60.]),scale,1.)
    report={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for w in WEIGHTS:
            name=f'reward_w{round(w*100):03d}';d=load(w)
            maps=np.load(ROOT/name/'validation_assay.npz')['donor_mappings']
            if args.limit:
                d={k:v[:args.limit] for k,v in d.items()}
                maps=derangements(args.limit,20,20260916)
            tasks=[(a[:150].astype(np.int64),r[:150].astype(float),scale) for a,r in zip(d['action'],d['reward'])]
            solutions=np.stack(list(pool.map(fit_one,tasks,chunksize=4)))
            best=solutions[np.arange(len(solutions)),solutions[:,:,4].argmin(1)]
            pa=best[:,:4]
            _,loss=soft_replay(d,d['reward'],pa,scale)
            assert np.allclose(loss[:,:150].sum(1),best[:,4],atol=1e-7)
            result,arrays=assay(d,pa,scale,maps)
            diag=dict(best_success_fraction=float(best[:,5].mean()),
                near_optimal_starts_mean=float((solutions[:,:,4]<=best[:,4,None]+1).sum(1).mean()),
                beta_upper_boundary_count=int(np.sum(pa[:,2]>5.999)),
                beta_lower_boundary_count=int(np.sum(pa[:,2]<-5.999)),
                mean_alpha=float(pa[:,0].mean()),mean_weight=float(pa[:,1].mean()),mean_beta=float(np.exp(pa[:,2]).mean()))
            report[name]=dict(groups=result,optimization=diag)
            np.savez_compressed(args.output/f'{name}.npz',**arrays,all_solutions=solutions,kernel_sign=d['kernel_sign'])
            (args.output/'summary.json').write_text(json.dumps(report,indent=2))
            print(name,diag,'heldout_nll',result['all']['intact_nll']['mean'],flush=True)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,test_used=False)))


if __name__=='__main__':main()
