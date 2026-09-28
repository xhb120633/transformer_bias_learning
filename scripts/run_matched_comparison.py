"""Original three-condition matched comparison; no training and no new-weight data."""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from fit_softmax_cognitive import fit_one, objective, soft_replay, BOUNDS
from run_weight_curve_a import replay, estimate
from mechcal.analysis.restless_recovery import load_split
from mechcal.analysis.generator_oracle_donor import _ordered_condition
from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.training.restless_dataset import RestlessTranscriptDataset

DATA=Path('outputs/restless_pooled_eps10_1000_20260827')
OUT=Path('outputs/matched_comparison_20260916')
CS=('reward_dominant','balanced','choice_dominant')
SEEDS=(11,22,33)


def nll(p,actions):
    return -np.log(np.take_along_axis(p,actions[...,None],axis=-1)[...,0])


def context(c):
    with np.load(f'outputs/centaur_donor_probabilities_test/{c}/trial_losses.npz') as z:
        ids=z['participants'].astype(str);maps=z['donor_mappings'].copy()
    assert len(ids)==len(set(ids))==250
    for m in maps:
        assert np.array_equal(np.sort(m),np.arange(250)) and np.all(m!=np.arange(250))
    d=_ordered_condition(load_split(DATA,'test'),c,ids)
    records=[json.loads(l) for l in Path(f'outputs/centaur_sft_eps10_seed11/{c}_test.jsonl').read_text().splitlines()]
    lookup={str(r['participant']):['ABCD'.index(x) for x in re.findall(r'<<([ABCD])>>',r['text'])] for r in records}
    assert np.array_equal(d['action'],np.asarray([lookup[i] for i in ids]))
    return d,maps,json.loads((DATA/'metadata.json').read_text())['q_pairwise_scale']


def fit():
    for c in CS:
        target=OUT/f'{c}_cognitive.npz'
        if target.exists():continue
        d,maps,scale=context(c)
        tasks=[(a[:150].astype(np.int64),r[:150].astype(float),scale) for a,r in zip(d['action'],d['reward'])]
        with ThreadPoolExecutor(max_workers=8) as pool:
            solutions=np.stack(list(pool.map(fit_one,tasks)))
        best=solutions[np.arange(250),solutions[:,:,4].argmin(1)].copy()
        original_best=best.copy()
        for i in np.flatnonzero(best[:,5]==0):
            aa,rr,_=tasks[i]
            opt=minimize(objective,best[i,:3],args=(aa,rr,scale,best[i,3]),jac=True,method='L-BFGS-B',bounds=BOUNDS,
                         options=dict(maxiter=2000,ftol=1e-11,gtol=1e-6,maxls=100))
            if opt.fun<=best[i,4]+1e-8:
                best[i]=[*opt.x,best[i,3],opt.fun,float(opt.success),opt.nit]
        pa=best[:,:4]
        p,loss=soft_replay(d,d['reward'],pa,scale)
        assert np.allclose(loss[:,:150].sum(1),best[:,4],atol=1e-7)
        op=replay(d,d['reward'],scale)
        assert np.allclose(op,d['choice_probability'],atol=1e-6)
        arrays=dict(participants=d['base_participant_id'],actions=d['action'],donor_mappings=maps,
                    kernel_sign=d['kernel_sign'],all_solutions=solutions,original_best=original_best,best=best,
                    intact_probabilities=p,intact_nll=loss,oracle_intact_probabilities=op)
        for local,name in ((False,'donor'),(True,'local')):
            ps,ls,ops=[],[],[]
            for m in maps:
                q,l=soft_replay(d,d['reward'][m],pa,scale,local)
                oq=replay(d,d['reward'][m],scale,local)
                assert np.allclose(q[:,0],p[:,0],atol=1e-12)
                if local:q,l,oq=q[:,199:200],l[:,199:200],oq[:,199:200]
                ps.append(q);ls.append(l);ops.append(oq)
            arrays[name+'_probabilities']=np.stack(ps)
            arrays[name+'_nll']=np.stack(ls)
            arrays['oracle_'+name+'_probabilities']=np.stack(ops)
        np.savez_compressed(target,**arrays)
        print('FIT_COMPLETE',c,'unconverged',int((best[:,5]==0).sum()),'beta_upper',int((pa[:,2]>5.999).sum()),flush=True)


@torch.no_grad()
def predict(model,tokens):
    result=[]
    for batch in tokens.split(64):
        logits=model(batch[:,:-1].cuda()).float()[:,0::2,1:5]
        result.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(result)


def export():
    torch.set_num_threads(4)
    for c in CS:
        d,maps,scale=context(c)
        ds=RestlessTranscriptDataset(DATA,'test',c)
        assert np.array_equal(ds.audit.base_participant_id,d['base_participant_id'])
        assert np.array_equal(ds.tokens[:,1::2].numpy()-1,d['action'])
        t=ds.tokens
        for seed in SEEDS:
            target=OUT/f'{c}_transformer_seed{seed}.npz'
            if target.exists():continue
            checkpoint=Path(f'outputs/transformer_eps10_large_seed{seed}_{c}/best.pt')
            ck=torch.load(checkpoint,weights_only=True,map_location='cpu')
            model=CausalTransformer(TransformerConfig(**ck['model_config'])).cuda().eval()
            model.load_state_dict(ck['model_state'])
            ip=predict(model,t);dp=[];lp=[]
            for m in maps:
                changed=t.clone();changed[:,2::2]=t[m,2::2]
                dp.append(predict(model,changed))
                changed=t.clone();changed[:,398]=t[m,398]
                lp.append(predict(model,changed)[:,199:200])
            dp=np.stack(dp);lp=np.stack(lp)
            assert np.allclose(dp[:,:,0],ip[None,:,0],atol=1e-6)
            np.savez_compressed(target,participants=d['base_participant_id'],actions=d['action'],
                donor_mappings=maps,intact_probabilities=ip,donor_probabilities=dp,local_probabilities=lp,
                checkpoint=np.asarray(str(checkpoint)),checkpoint_epoch=ck['epoch'])
            print('EXPORT_COMPLETE',c,seed,flush=True)
            del model
            torch.cuda.empty_cache()


def norm(p):
    p=p.astype(float)
    assert np.isfinite(p).all() and np.all(p>=0) and np.all(p.sum(-1)>0)
    return p/p.sum(-1,keepdims=True)


def metrics(ip,dp,lp,actions,oi,od,ol,exact_losses=None):
    ip,dp,lp=norm(ip),norm(dp),norm(lp)
    il=nll(ip,actions) if exact_losses is None else exact_losses[0]
    dl=nll(dp,actions[None]) if exact_losses is None else exact_losses[1]
    ll=nll(lp,actions[None,:,199:200]) if exact_losses is None else exact_losses[2]
    return dict(intact_nll=il[:,150:].mean(1),donor_delta_nll=(dl[:,:,150:]-il[None,:,150:]).mean((0,2)),
                donor200_delta_nll=(dl[:,:,199:200]-il[None,:,199:200]).mean((0,2)),
                donor200_response_error=(.5*np.abs((dp-ip[None])-(od-oi[None])).sum(-1))[:,:,199:200].mean((0,2)),
                local200_delta_nll=(ll-il[None,:,199:200]).mean((0,2)),
                donor_tv=(.5*np.abs(dp-ip[None]).sum(-1))[:,:,150:].mean((0,2)),
                donor_response_error=(.5*np.abs((dp-ip[None])-(od-oi[None])).sum(-1))[:,:,150:].mean((0,2)),
                local200_response_error=(.5*np.abs((lp-ip[None,:,199:200])-(ol-oi[None,:,199:200])).sum(-1)).mean((0,2)))


def summarize():
    summary={};vectors={}
    for c in CS:
        d,maps,scale=context(c);a=d['action'];ids=d['base_participant_id']
        z=dict(np.load(OUT/f'{c}_cognitive.npz'))
        assert np.array_equal(ids,z['participants']) and np.array_equal(a,z['actions'])
        oi=z['oracle_intact_probabilities'];od=z['oracle_donor_probabilities'];ol=z['oracle_local_probabilities']
        models={'Oracle':[metrics(oi,od,ol,a,oi,od,ol)],
                'Cognitive softmax':[metrics(z['intact_probabilities'],z['donor_probabilities'],z['local_probabilities'],a,oi,od,ol,
                    (z['intact_nll'],z['donor_nll'],z['local_nll']))]}
        for family in ('GRU','Transformer','Llama'):
            runs=[]
            for seed in ((11,) if family=='Llama' else SEEDS):
                if family=='Transformer':
                    q=dict(np.load(OUT/f'{c}_transformer_seed{seed}.npz'))
                    ip,dp,lp=[q[k+'_probabilities'] for k in ('intact','donor','local')]
                else:
                    path=(f'outputs/gru_suite_20260907/{c}_full_seed{seed}/predictions.npz' if family=='GRU' else f'outputs/centaur_donor_probabilities_test/{c}/trial_losses.npz')
                    q=dict(np.load(path));ip=q['intact_choice_probabilities'];dp=q['donor_choice_probabilities']
                    if family=='GRU':
                        lp=q['local_choice_probabilities'][:,:,np.flatnonzero(q['trial_indices']==200)]
                    else:
                        with np.load(f'outputs/centaur_local_reward_test/{c}/trial_losses.npz') as loc:
                            assert np.array_equal(loc['participants'],ids) and np.array_equal(loc['donor_mappings'],maps)
                            ix=np.flatnonzero(loc['trial_indices']==200)
                            assert np.allclose(norm(loc['intact_choice_probabilities'][:,ix]),norm(ip[:,199:200]),atol=1e-6)
                            lp=loc['donor_choice_probabilities'][:,:,ix]
                assert np.array_equal(q['participants'],ids) and np.array_equal(q['donor_mappings'],maps)
                if 'actions' in q: assert np.array_equal(q['actions'],a)
                assert dp.shape==(20,250,200,4) and lp.shape==(20,250,1,4)
                v=metrics(ip,dp,lp,a,oi,od,ol)
                co_path=(f'outputs/transformer_choice_only_20260915/{c}_seed{seed}/predictions.npz' if family=='Transformer'
                         else f'outputs/gru_suite_20260907/{c}_choice_only_seed{seed}/predictions.npz' if family=='GRU'
                         else f'outputs/section_a_llama_probabilities_20260915/{c}/trial_losses.npz')
                with np.load(co_path) as co:
                    order=np.array([list(co['participants'].astype(str)).index(str(i)) for i in ids])
                    assert np.array_equal(co['actions'][order],a)
                    cp=norm(co['intact_choice_probabilities'][order])
                v['choice_only_nll']=nll(cp,a)[:,150:].mean(1)
                v['training_ablation_delta_nll']=v['choice_only_nll']-v['intact_nll']
                runs.append(v)
            models[family]=runs
        summary[c]={}
        for family,runs in models.items():
            mean={k:np.mean([r[k] for r in runs],0) for k in runs[0]}
            summary[c][family]=dict(n_runs=len(runs),run_means=[{k:float(v.mean()) for k,v in r.items()} for r in runs],strata={})
            for group,mask in [('all',np.ones(250,bool)),('positive',d['kernel_sign']>0),('negative',d['kernel_sign']<0)]:
                summary[c][family]['strata'][group]={k:estimate(v,mask) for k,v in mean.items()}
            for k,v in mean.items():vectors[f'{c}__{family}__{k}']=v
        for family in ('GRU','Transformer','Llama'):
            summary[c][family]['paired_vs_cognitive']={}
            stable=z['best'][:,2]<5.999
            summary[c][family]['excluding_cognitive_beta_upper_bound']={'n_retained':int(stable.sum())}
            for key in ('intact_nll','donor_response_error','donor200_response_error','local200_response_error'):
                diff=vectors[f'{c}__{family}__{key}']-vectors[f'{c}__Cognitive softmax__{key}']
                summary[c][family]['paired_vs_cognitive'][key]=estimate(diff,np.ones(250,bool))
                summary[c][family]['excluding_cognitive_beta_upper_bound'][key]=estimate(diff,stable)
        vectors[c+'__participants']=ids;vectors[c+'__kernel_sign']=d['kernel_sign']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(OUT/'participant_vectors.npz',**vectors)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,matched_participants=True,matched_maps=True,
        n_subjects_per_condition=250,donor_repeats=20,donor_trials='151-200',local_trials=[200],neural_retraining=False)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['fit','export','summarize']);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {'fit':fit,'export':export,'summarize':summarize}[args.phase]()
