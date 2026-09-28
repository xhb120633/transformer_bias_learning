"""Task2 A/B validation analysis. No test access; matched observable histories."""
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from mechcal.generators.spatial_mixture import SpatialConfig, replay
from mechcal.training.spatial_transcripts import encode_map, CHOICE_IDS
from mechcal.models.gru import CausalGRU, GRUConfig
from mechcal.models.causal_transformer import CausalTransformer, TransformerConfig
from ablate_spatial_references import nll, summarize

ROOT=Path('outputs/spatial_pilot_v1')
OUT=Path('outputs/spatial_neural_analysis_v1')
WEIGHTS=[0.,.1,.3,.5,.7,.9,1.]
SEEDS=[11,22,33]


def fit_one(task):
    import recover_spatial_pilot as recovery
    recovery.MODELS=('mixture',)
    return recovery.fit_subject(task)


def predict(model, episodes, mode, max_length=None):
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(episodes),32):
            encoded=[encode_map(ep,mode) for ep in episodes[start:start+32]]
            # Final reward is after the last supervised choice; trim irrelevant
            # suffix, never a choice or a preceding reward. Handles longer donors.
            needed=max(max(i for i,v in enumerate(m) if v)+1 for s,m in encoded)
            length=(max_length+1) if max_length is not None else (512 if mode=='full' else 128)
            if needed>length:
                raise ValueError(f'Donor prefix {needed} exceeds evaluation capacity {length}')
            # Fix both dimensions for intact/donor calls: cuDNN rounding can
            # otherwise change identical prefixes slightly across batch shapes.
            x=torch.zeros((32,length),dtype=torch.long,device='cuda')
            mask=torch.zeros_like(x,dtype=torch.bool)
            for i,(s,m) in enumerate(encoded):
                stop=min(len(s),length)
                x[i,:stop]=torch.tensor(s[:stop],device='cuda')
                mask[i,:stop]=torch.tensor(m[:stop],device='cuda')
            logits=model(x[:,:-1])[mask[:,1:]][:,CHOICE_IDS].float()
            predictions.append(torch.softmax(logits,-1).cpu().numpy().reshape(len(encoded),20,25))
    result=np.concatenate(predictions)
    assert np.isfinite(result).all() and (result>0).all()
    np.testing.assert_allclose(result.sum(-1),1,atol=2e-6)
    return result


def load_model(family,mode,name,seed):
    path=Path('outputs/spatial_neural_v1')/f'{family}_{mode}_{name}_seed{seed}'/'best.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['config']['mode']==mode and ck['config']['seed']==seed
    cfg=ck['config']['model']
    model=CausalGRU(GRUConfig(**cfg)) if family=='gru' else CausalTransformer(TransformerConfig(**cfg))
    model.load_state_dict(ck['model']);model=model.cuda().eval()
    return model,cfg.get('max_sequence_length'),hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text());c=SpatialConfig(**protocol['config'])
    rng=np.random.default_rng(81073);donors=[]
    for _ in range(20):
        while True:
            m=rng.permutation(50)
            if np.all(m!=np.arange(50)):break
        donors.append(m)
    donors=np.array(donors)
    records={};tasks=[]
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}'
        rows=[json.loads(s) for s in (ROOT/name/'val_observable.jsonl').read_text().splitlines()]
        assert len(rows)==50
        records[name]=rows
        tasks.extend((w,r,protocol['config'],31415+i) for i,r in enumerate(rows))
    fitpath=OUT/'fits.json'
    if not fitpath.exists():
        fits=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for row in pool.map(fit_one,tasks):
                fits.append(row)
                print('FIT',len(fits),350,flush=True)
        fitpath.write_text(json.dumps(fits,indent=2))
    else:
        fits=json.loads(fitpath.read_text())
    assert len(fits)==350
    failures=[(r['weight'],r['participant']) for r in fits if not r['models']['mixture']['optimizer_success']]
    if failures:raise RuntimeError(f'Inspect failed fits before analysis: {failures}')
    summary={};hashes={}
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}';rows=records[name]
        with np.load(ROOT/name/'val_audit.npz',allow_pickle=False) as z:
            d={k:z[k] for k in ('actions','rewards','cue_arm','cue_reward','participant','gp_length','reward_temperature','local_temperature','probabilities')}
        assert [r['participant'] for r in rows]==d['participant'].tolist()
        a=d['actions'];all_eps=[ep for r in rows for ep in r['rounds']]
        held_eps=[ep for r in rows for ep in r['rounds'][6:]]
        s={};saved={'participants':d['participant'],'actions':a,'donor_maps':donors}
        p=d['probabilities']
        s['oracle_full_A_nll']=summarize(nll(p,a).mean((1,2)))
        s['oracle_full_A_accuracy']=summarize((p.argmax(-1)==a).mean((1,2)))
        for ref in ('oracle','fit'):
            intact=np.empty((50,2,20,25));abl=np.empty((20,50,2,20,25))
            for i,row in enumerate(rows):
                if ref=='oracle':
                    pars={k:float(d[k][i]) for k in ('gp_length','reward_temperature','local_temperature')};weight=w
                else:
                    fit=next(f for f in fits if f['weight']==w and f['participant']==row['participant'])
                    pars=fit['models']['mixture']['parameters'].copy();weight=pars.pop('weight')
                for j,b in enumerate((6,7)):
                    args=(c,pars,weight,int(d['cue_arm'][i,b]),float(d['cue_reward'][i,b]),a[i,b])
                    intact[i,j]=replay(*args,d['rewards'][i,b])
                    for k,m in enumerate(donors):abl[k,i,j]=replay(*args,d['rewards'][m[i],b])
            if ref=='oracle':np.testing.assert_allclose(intact,p[:,6:],atol=1e-12)
            np.testing.assert_allclose(abl[:,:,:,0],np.broadcast_to(intact[None,:,:,0],abl[:,:,:,0].shape),atol=1e-12)
            if ref=='oracle' and w==0:np.testing.assert_array_equal(abl,np.broadcast_to(intact,abl.shape))
            base=nll(intact,a[:,6:]).mean((1,2));change=nll(abl,a[None,:,6:]).mean((0,2,3))-base
            saved[ref+'_B_intact']=intact;saved[ref+'_B_donor']=abl
            for suffix,values in [('intact_nll',base),('donor_nll',base+change),('delta_nll',change),('relative_pct',100*change/base)]:
                s[ref+'_B_'+suffix]=summarize(values)
            saved[ref+'_B_delta_nll']=change
        for family in ('gru','transformer'):
            for mode in ('full','choice_only'):
                pa=[];qb=[]
                for seed in SEEDS:
                    cache=OUT/f'{name}_{family}_{mode}_seed{seed}_fixedshape.npz'
                    if cache.exists():
                        with np.load(cache) as z:
                            pa.append(z['intact']);hashes[cache.stem]=str(z['checkpoint_sha256'])
                            if mode=='full':qb.append(z['donor'])
                        continue
                    model,capacity,sha=load_model(family,mode,name,seed)
                    intact=predict(model,all_eps,mode,capacity).reshape(50,8,20,25)
                    export={'intact':intact,'checkpoint_sha256':sha}
                    if mode=='full':
                        q=[]
                        for mapping in donors:
                            altered=[]
                            for i in range(50):
                                for b in (6,7):
                                    ep=copy.deepcopy(rows[i]['rounds'][b])
                                    ep['rewards']=rows[int(mapping[i])]['rounds'][b]['rewards']
                                    altered.append(ep)
                            q.append(predict(model,altered,mode,capacity).reshape(50,2,20,25))
                        q=np.array(q)
                        np.testing.assert_allclose(q[:,:,:,0],np.broadcast_to(intact[None,:,6:,0],q[:,:,:,0].shape),atol=2e-5,rtol=2e-5)
                        export['donor']=q;qb.append(q)
                    np.savez_compressed(cache,**export);hashes[cache.stem]=sha;pa.append(intact)
                    del model;torch.cuda.empty_cache()
                    print('NEURAL',name,family,mode,seed,flush=True)
                pa=np.array(pa)
                per_seed_nll=nll(pa,a[None]).mean((2,3))
                per_seed_acc=(pa.argmax(-1)==a[None]).mean((2,3))
                key=family+'_'+mode
                s[key+'_A_nll']=summarize(per_seed_nll.mean(0))
                s[key+'_A_nll']['seed_means']=per_seed_nll.mean(1).tolist()
                s[key+'_A_accuracy']=summarize(per_seed_acc.mean(0))
                saved[key+'_A_subject_nll']=per_seed_nll
                if mode=='full':
                    qb=np.array(qb)
                    base=nll(pa[:,:,6:],a[None,:,6:]).mean((2,3))
                    change=nll(qb,a[None,None,:,6:]).mean((1,3,4))-base
                    saved[family+'_B_delta_nll']=change
                    for suffix,values in [('intact_nll',base),('donor_nll',base+change),('delta_nll',change),('relative_pct',100*change/base)]:
                        s[family+'_B_'+suffix]=summarize(values.mean(0))
                        s[family+'_B_'+suffix]['seed_means']=values.mean(1).tolist()
        summary[str(w)]=s
        np.savez_compressed(OUT/(name+'_matched.npz'),**saved)
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        print('WEIGHT_COMPLETE',w,flush=True)
    (OUT/'protocol.json').write_text(json.dumps(dict(
        split='validation; test untouched',subjects=50,weights=WEIGHTS,seeds=SEEDS,donors=20,
        A='All8 maps: neural full/choice-only; full-information oracle; uniform random',
        B='Same last2 maps for all models; first6 maps used only for individual generator refits',
        fit='Known generator family/noise/lapse; mixture parameters including w fitted, never given GT; 3-start L-BFGS-B',
        caveat='Refit uses participant calibration maps; neural models do not. Validation selected neural checkpoints, not independent test.',
        intervention='Cumulative donor chosen rewards; actual choices and initial cue preserved; map reset',
        ci='2000 participant bootstraps of seed-averaged metrics; seed means separately reported; donors held fixed',
        checkpoint_sha256=hashes,optimizer_failures=failures),indent=2))
    plot(summary)
    (OUT/'ANALYSIS_COMPLETE').write_text('Validation diagnostic A/B complete; test untouched.\n')


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'oracle':('Oracle','#444444'),'fit':('Fitted generator','#b87826'),
            'gru':('GRU','#228b78'),'transformer':('Transformer','#536eba')}
    def curve(ax,prefix,suffix):
        key=prefix+'_'+suffix;label,color=styles[prefix.split('_')[0]]
        values=[summary[str(w)][key] for w in WEIGHTS]
        y=np.array([v['mean'] for v in values]);ci=np.array([v['ci95'] for v in values])
        if prefix=='oracle_full':label='Full-information oracle'
        ax.plot(WEIGHTS,y,'o-',label=label,color=color,ms=4)
        ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],color=color,alpha=.10)
        if 'seed_means' in values[0]:
            seeds=np.array([v['seed_means'] for v in values])
            for i in range(3):ax.plot(WEIGHTS,seeds[:,i],color=color,alpha=.22,lw=.7)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    for ax,mode,title in zip(axes,('full','choice_only'),('A | Full-input training','A | Choice-only training')):
        for family in ('gru','transformer'):curve(ax,family+'_'+mode,'A_nll')
        curve(ax,'oracle_full','A_nll')
        ax.axhline(np.log(25),color='#aaa',ls=':',label='Uniform random')
        ax.set(title=title,xlabel='Generating reward weight',ylabel='Validation choice NLL (nats)')
    axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'A_prediction.png',dpi=180);fig.savefig(OUT/'A_prediction.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.8))
    for ax,suffix,title in zip(axes,('B_intact_nll','B_donor_nll','B_delta_nll'),('B1 | Original rewards','B2 | Donor-replaced rewards','B3 | Ablation effect')):
        for prefix in styles:curve(ax,prefix,suffix)
        ax.set(title=title,xlabel='Generating reward weight',ylabel='Delta NLL (nats / choice)' if 'delta' in suffix else 'Validation NLL (nats / choice)')
        if 'delta' in suffix:ax.axhline(0,color='#aaa',lw=.8)
        else:ax.axhline(np.log(25),color='#aaa',ls=':',label='Uniform random')
    # Identical absolute-NLL scales make before/after comparisons interpretable.
    low=min(axes[0].get_ylim()[0],axes[1].get_ylim()[0])
    high=max(axes[0].get_ylim()[1],axes[1].get_ylim()[1])
    for ax in axes[:2]:ax.set_ylim(low,high)
    axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'B_donor.png',dpi=180);fig.savefig(OUT/'B_donor.pdf');plt.close(fig)


def add_donor_summaries():
    """Reaggregate saved predictions; no model evaluation or new data access."""
    summary=json.loads((OUT/'summary.json').read_text())
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}';s=summary[str(w)]
        with np.load(OUT/(name+'_matched.npz')) as z:
            actions=z['actions'][:,6:]
            for ref in ('oracle','fit'):
                values=nll(z[ref+'_B_donor'],actions[None]).mean((0,2,3))
                s[ref+'_B_donor_nll']=summarize(values)
        for family in ('gru','transformer'):
            values=[]
            for seed in SEEDS:
                with np.load(OUT/f'{name}_{family}_full_seed{seed}_fixedshape.npz') as z:
                    values.append(nll(z['donor'],actions[None]).mean((0,2,3)))
            values=np.array(values)
            s[family+'_B_donor_nll']=summarize(values.mean(0))
            s[family+'_B_donor_nll']['seed_means']=values.mean(1).tolist()
        for model in ('oracle','fit','gru','transformer'):
            np.testing.assert_allclose(s[model+'_B_donor_nll']['mean']-s[model+'_B_intact_nll']['mean'],
                                       s[model+'_B_delta_nll']['mean'],atol=1e-6)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plot(summary)


def preliminary_a():
    """A is available immediately from completed held-out prediction metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary={}
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}';s={}
        with np.load(ROOT/name/'val_audit.npz') as z:
            s['oracle_nll']=float(nll(z['probabilities'],z['actions']).mean())
        for family in ('gru','transformer'):
            for mode in ('full','choice_only'):
                s[family+'_'+mode]=[json.loads((Path('outputs/spatial_neural_v1')/f'{family}_{mode}_{name}_seed{seed}'/'metrics.json').read_text())['val']['choice_nll'] for seed in SEEDS]
        summary[str(w)]=s
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,mode,title in zip(axes,('full','choice_only'),('Full-input training','Choice-only training')):
        for family,color in [('gru','#228b78'),('transformer','#536eba')]:
            values=np.array([summary[str(w)][family+'_'+mode] for w in WEIGHTS])
            ax.plot(WEIGHTS,values.mean(1),'o-',label='GRU' if family=='gru' else 'Transformer',color=color,ms=4)
            for i in range(3):ax.plot(WEIGHTS,values[:,i],color=color,alpha=.25,lw=.8)
        ax.plot(WEIGHTS,[summary[str(w)]['oracle_nll'] for w in WEIGHTS],'o-',color='#444',label='Full-information oracle',ms=4)
        ax.axhline(np.log(25),ls=':',color='#999',label='Uniform random')
        ax.set(xlabel='Generating reward weight',title=title,ylabel='Validation choice NLL (nats)')
    axes[0].legend(frameon=False,fontsize=8)
    fig.text(.5,.01,'50 validation participants; 8 maps each. Thin lines: 3 training seeds. Oracle retains reward information.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(OUT/'A_preliminary.png',dpi=180);fig.savefig(OUT/'A_preliminary.pdf');plt.close(fig)
    (OUT/'A_preliminary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
