"""Development-only A/B pairing, cross-fitted calibration and oracle diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.models.gru import CausalGRU, GRUConfig
from analyze_trainfit_baselines import fit as fit_baselines, predict as baseline_predict
from run_reference_followup import ROOT, WEIGHTS, load, params
from run_weight_curve_a import replay, estimate

OUT=Path('outputs/task1_diagnostics_20260916')
FULL=Path('outputs/weight_neural_full_20260916')
CO=Path('outputs/weight_neural_choice_only_20260916')
MODES=('donor','local','suffix_donor')
SEEDS=(11,22,33)

def loss(p,a):
    return -np.log(np.take_along_axis(p,a[...,None],-1)[...,0])

def temper(p,t):
    lp=np.log(np.clip(p,1e-30,1))/t
    return np.exp(lp-logsumexp(lp,axis=-1,keepdims=True))

def calibrate(p,a,folds):
    """One temperature per training fold, selected only on intact probabilities."""
    result=np.empty_like(p,dtype=float);ts=[]
    for k in range(5):
        train=folds!=k;val=~train
        fit=minimize_scalar(lambda x:loss(temper(p[train],np.exp(x)),a[train]).mean(),
                            bounds=(np.log(.05),np.log(20)),method='bounded')
        assert fit.success
        t=float(np.exp(fit.x));ts.append(t);result[val]=temper(p[val],t)
    return result,ts

def response_metrics(p,q,op,oq):
    d=q-p[None];o=oq-op[None]
    active=(np.abs(o).sum(-1)>1e-10)
    dot=(d*o).sum(-1)
    oe=(o*o).sum(-1);me=(d*d).sum(-1)*active
    oe=oe.sum((0,2));me=me.sum((0,2));dot=dot.sum((0,2))
    gain=np.divide(dot,oe,out=np.full_like(dot,np.nan),where=oe>0)
    cosine=np.divide(dot,np.sqrt(oe*me),out=np.full_like(dot,np.nan),where=(oe*me)>0)
    return dict(response_error=(.5*np.abs(d-o).sum(-1)).mean((0,2)),
                response_tv=(.5*np.abs(d).sum(-1)).mean((0,2)),
                projection_gain=gain,response_cosine=cosine,
                oracle_active_fraction=active.mean((0,2)))

def summarize(v,sign):
    result={}
    for g,mask in [('all',np.ones(len(sign),bool)),('positive',sign>0),('negative',sign<0)]:
        result[g]={}
        for k,x in v.items():
            valid=mask&np.isfinite(x)
            result[g][k]=dict(estimate(x,valid),n=int(valid.sum())) if valid.any() else dict(mean=None,ci95=None,n=0)
    return result

@torch.no_grad()
def export_choice():
    torch.set_num_threads(4)
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}';d=load(w)
        tok=torch.tensor(np.concatenate([d['tokens'][:,:1],d['tokens'][:,1::2]],1).astype(np.int64))
        assert np.array_equal(tok[:,1:].numpy()-1,d['action'])
        for family in ('gru','transformer'):
            for seed in SEEDS:
                target=OUT/f'{family}_{name}_seed{seed}_choice.npz'
                if target.exists():continue
                run=CO/f'{family}_{name}_choice_only_seed{seed}'
                assert (run/'metrics.json').exists()
                ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
                model=(CausalGRU(GRUConfig(**ck['model_config'])) if family=='gru' else CausalTransformer(TransformerConfig(**ck['model_config']))).cuda().eval()
                model.load_state_dict(ck['model_state'])
                p=np.concatenate([model(b[:,:-1].cuda()).float()[...,1:5].softmax(-1).cpu().numpy() for b in tok.split(32)])
                assert p.shape==(250,200,4)
                # Predictions through choice 100 cannot depend on later tokens.
                assert np.allclose(p[:2,:100],model(tok[:2,:100].cuda()).float()[...,1:5].softmax(-1).cpu().numpy(),atol=2e-5)
                np.savez_compressed(target,participants=d['base_participant_id'],probabilities=p)
                del model;torch.cuda.empty_cache()
        print('CHOICE_EXPORTED',w,flush=True)

def analyze():
    scale=json.loads((ROOT/'protocol.json').read_text())['q_scale']
    report={};baseline_specs={};temperatures={}
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}';d=load(w);a=d['action'][:,150:]
        train=dict(np.load(ROOT/name/'train_000.npz'))
        assert not set(train['base_participant_id'])&set(d['base_participant_id'])
        fitted=fit_baselines(train['action']);baseline_specs[str(w)]=fitted
        bp=baseline_predict(d['action'],fitted)
        changed=d['action'].copy();changed[:,175:]=(changed[:,175:]+1)%4
        for k,x in bp.items():assert np.array_equal(x[:,:175],baseline_predict(changed,fitted)[k][:,:175])
        vectors={}
        for k,x in bp.items():
            p=x[:,149:];vectors['baseline_'+k+'_nll']=loss(p,a).mean(1)
            vectors['baseline_'+k+'_accuracy']=(p.argmax(-1)==a).mean(1)
        maps=np.load(ROOT/name/'validation_assay.npz')['donor_mappings']
        op=replay(d,d['reward'],scale)[:,150:];ob=loss(op,a)
        vectors['oracle_nll']=ob.mean(1);oracle_changed={}
        for mode in MODES:
            qs=[]
            for m in maps:
                r=d['reward'][m].copy()
                if mode=='suffix_donor':r[:,:150]=d['reward'][:,:150]
                qs.append(replay(d,r,scale,mode=='local')[:,150:])
            oq=np.stack(qs);oracle_changed[mode]=oq
            active=(np.abs(oq-op[None]).sum(-1)>1e-10)
            kl=(op[None]*np.log(op[None]/oq)).sum(-1)
            empirical=loss(oq,a[None])-ob[None]
            vectors['oracle_'+mode+'_winner_set_change']=active.mean((0,2))
            vectors['oracle_'+mode+'_expected_kl']=kl.mean((0,2))
            vectors['oracle_'+mode+'_delta_nll']=empirical.mean((0,2))
            vectors['oracle_'+mode+'_tv']=(.5*np.abs(oq-op[None]).sum(-1)).mean((0,2))
            # For unique-winner pairs, KL = .9 log(37) iff winner changes.
            unique=((op==op.max(-1,keepdims=True)).sum(-1)==1)[None]&((oq==oq.max(-1,keepdims=True)).sum(-1)==1)
            assert np.allclose(kl[unique],(.9*np.log(37)*active)[unique],atol=1e-10)
        folds=np.empty(250,int);folds[np.random.default_rng(20260916).permutation(250)]=np.arange(250)%5
        for family in ('gru','transformer'):
            runs=[]
            for seed in SEEDS:
                z=dict(np.load(FULL/f'{family}_{name}_full_seed{seed}'/'assay_full.npz'))
                co=dict(np.load(OUT/f'{family}_{name}_seed{seed}_choice.npz'))
                assert np.array_equal(z['participants'],co['participants']) and np.array_equal(z['donor_mappings'],maps)
                p=z['intact_probabilities'][:,150:].astype(float);p/=p.sum(-1,keepdims=True)
                cp=co['probabilities'][:,150:].astype(float);cp/=cp.sum(-1,keepdims=True)
                calp,temps=calibrate(p,a,folds);temperatures[f'{w}_{family}_{seed}']=temps
                bl=loss(p,a);cb=loss(calp,a)
                v=dict(full_nll=bl.mean(1),choice_nll=loss(cp,a).mean(1),
                    full_accuracy=(p.argmax(-1)==a).mean(1),choice_accuracy=(cp.argmax(-1)==a).mean(1),
                    reward_predictive_gain=(loss(cp,a)-bl).mean(1),calibrated_full_nll=cb.mean(1))
                for mode in MODES:
                    q=z[mode+'_probabilities'].astype(float);q/=q.sum(-1,keepdims=True)
                    cq=np.empty_like(q)
                    for k,t in enumerate(temps):cq[:,folds==k]=temper(q[:,folds==k],t)
                    for label,pp,qq,ll in [('raw',p,q,bl),('calibrated',calp,cq,cb)]:
                        v[label+'_'+mode+'_delta_nll']=(loss(qq,a[None])-ll[None]).mean((0,2))
                        for k,x in response_metrics(pp,qq,op,oracle_changed[mode]).items():v[label+'_'+mode+'_'+k]=x
                    v[mode+'_ablation_minus_retraining_gap']=v['raw_'+mode+'_delta_nll']-v['reward_predictive_gain']
                v['choice_minus_train_lag_nll']=v['choice_nll']-vectors['baseline_train_selected_lag_nll']
                runs.append(v)
            for k in runs[0]:
                arr=np.stack([r[k] for r in runs]);vectors[family+'_'+k]=arr.mean(0)
        np.savez_compressed(OUT/f'{name}_diagnostics.npz',participants=d['base_participant_id'],sign=d['kernel_sign'],folds=folds,**vectors)
        report[str(w)]=summarize(vectors,d['kernel_sign'])
        print('DIAGNOSTICS',w, 'lag',fitted['selected']['lag'],flush=True)
        (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'baseline_fits.json').write_text(json.dumps(baseline_specs,indent=2))
    (OUT/'temperatures.json').write_text(json.dumps(temperatures,indent=2))

def common():
    scale=json.loads((ROOT/'protocol.json').read_text())['q_scale'];report={}
    for w in WEIGHTS:
        collected={}
        for source in WEIGHTS:
            d=load(source);maps=np.load(ROOT/f'reward_w{round(source*100):03d}'/'validation_assay.npz')['donor_mappings']
            changed=params(d,d['alpha'],np.full(250,w),d['kernel_sign'])
            p=replay(changed,d['reward'],scale)[:,150:]
            for mode in MODES:
                kl=[];events=[]
                for m in maps:
                    r=d['reward'][m].copy()
                    if mode=='suffix_donor':r[:,:150]=d['reward'][:,:150]
                    q=replay(changed,r,scale,mode=='local')[:,150:]
                    kl.append((p*np.log(p/q)).sum(-1).mean(1))
                    events.append((np.abs(q-p).sum(-1)>1e-10).mean(1))
                collected.setdefault(mode+'_expected_kl',[]).append(np.mean(kl,0))
                collected.setdefault(mode+'_winner_set_change',[]).append(np.mean(events,0))
        vectors={k:np.mean(v,0) for k,v in collected.items()}
        report[str(w)]=summarize(vectors,d['kernel_sign'])
        np.savez_compressed(OUT/f'common_w{round(w*100):03d}.npz',**vectors)
        (OUT/'common_history.json').write_text(json.dumps(report,indent=2))
        print('COMMON_SUFFIX',w,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['export','analyze','common']);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {'export':export_choice,'analyze':analyze,'common':common}[args.phase]()
