"""Matched cumulative donor ablation for oracle and individually fitted generator."""
import json
from pathlib import Path
import numpy as np
from mechcal.generators.spatial_mixture import SpatialConfig,replay


def nll(p,actions):
    return -np.log(np.take_along_axis(p,actions[...,None],-1)[...,0])


def summarize(x):
    x=np.asarray(x);rng=np.random.default_rng(731)
    means=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist())


def main():
    root=Path('outputs/spatial_pilot_v1');out=Path('outputs/spatial_reference_ablation_v1');out.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text());c=SpatialConfig(**protocol['config'])
    fits=json.loads(Path('outputs/spatial_recovery_pilot_v1/results.json').read_text())
    weights=protocol['weights'];n=10;repeats=20;rng=np.random.default_rng(81073);maps=[]
    for _ in range(repeats):
        while True:
            m=rng.permutation(n)
            if np.all(m!=np.arange(n)):break
        maps.append(m)
    maps=np.array(maps);summary={}
    for w in weights:
        name=f'reward_w{round(w*100):03d}'
        with np.load(root/name/'train_audit.npz') as z:d={k:v[:n] for k,v in z.items()}
        actions=d['actions'][:,6:];reward=d['rewards'][:,6:]
        p=np.empty((2,n,2,c.trials,25));q=np.empty((2,repeats,n,2,c.trials,25))
        for i in range(n):
            row=next(r for r in fits if r['weight']==w and r['participant']==str(d['participant'][i]))
            true={k:float(d[k][i]) for k in ('gp_length','reward_temperature','local_temperature')}
            estimated=row['models']['mixture']['parameters'].copy();fitted_w=estimated.pop('weight')
            for model,(pars,weight) in enumerate(((true,w),(estimated,fitted_w))):
                for b in range(2):
                    args=(c,pars,weight,int(d['cue_arm'][i,b+6]),float(d['cue_reward'][i,b+6]),actions[i,b])
                    p[model,i,b]=replay(*args,reward[i,b])
                    for j,m in enumerate(maps):q[model,j,i,b]=replay(*args,reward[m[i],b])
        # No past chosen reward at first choice: cue is preserved in this assay.
        np.testing.assert_allclose(q[:,:,:,:,0],p[:,None,:,:,0]+np.zeros_like(q[:,:,:,:,0]),atol=1e-12)
        if w==0:np.testing.assert_allclose(q[0],np.broadcast_to(p[0],q[0].shape),atol=0,rtol=0)
        vectors={}
        for mi,label in enumerate(('oracle','fit')):
            base=nll(p[mi],actions).mean((1,2))
            ablated=nll(q[mi],actions[None]).mean((0,2,3))
            kl=(p[mi][None]*(np.log(p[mi][None])-np.log(q[mi]))).sum(-1).mean((0,2,3))
            vectors[label+'_intact_nll']=base;vectors[label+'_donor_nll']=ablated
            vectors[label+'_delta_nll']=ablated-base
            vectors[label+'_relative_delta_pct']=100*(ablated-base)/base
            vectors[label+'_expected_kl']=kl
        change=q-p[:,None]
        vectors['response_disagreement']=.5*np.abs(change[1]-change[0]).sum(-1).mean((0,2,3))
        vectors['delta_difference']=vectors['fit_delta_nll']-vectors['oracle_delta_nll']
        summary[str(w)]={k:summarize(v) for k,v in vectors.items()}
        np.savez_compressed(out/(name+'.npz'),**vectors,participants=d['participant'],donor_maps=maps,
                            actions=actions,intact_probabilities=p,donor_probabilities=q)
        print('ABLATION',w,summary[str(w)]['oracle_delta_nll']['mean'],summary[str(w)]['fit_delta_nll']['mean'],flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest=dict(subjects_per_weight=n,repeats=repeats,split='TRAIN participants, maps 7-8 only',
        fitting='Mixture model for every weight; parameters fit only on maps 1-6, never refit after ablation',
        intervention='All chosen rewards replaced by same-round/trial donor rewards; initial cue unchanged',
        choices='Actual choices held fixed, standard one-step-ahead replay, state resets every map',
        pairing='Same 20 derangements across weights/models. Ten participants paired across weights.',
        uncertainty='Participant bootstrap after averaging donor maps and trials; n=10 exploratory intervals',
        caveat='Known family/noise/lapse. This is not reward removal or a choice-only retraining baseline.',
        test_or_validation_used=False)
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for mi,(title,metric) in enumerate([('Intact prediction','intact_nll'),('Donor-reward prediction','donor_nll'),('Ablation effect','delta_nll')]):
        for label,color,ls in [('oracle','#424b57','-'),('fit','#b97922','--')]:
            vals=[summary[str(w)][label+'_'+metric] for w in weights];y=[v['mean'] for v in vals];ci=np.array([v['ci95'] for v in vals])
            axs[mi].plot(weights,y,ls+'o',color=color,label='Oracle' if label=='oracle' else 'Fitted generator',ms=4)
            axs[mi].fill_between(weights,ci[:,0],ci[:,1],color=color,alpha=.12)
        if mi<2:axs[mi].axhline(np.log(25),color='#aaa',ls=':',label='Uniform random')
        else:axs[mi].axhline(0,color='#aaa',lw=.8)
        axs[mi].set(xlabel='Generating reward weight',ylabel='Nats / choice',title=title)
    axs[0].legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(out/'references.png',dpi=180);fig.savefig(out/'references.pdf');plt.close(fig)
    lines=['# Spatial oracle and fitted-generator donor ablation','',
           '10 paired subjects/weight, fit first 6 maps, evaluate last 2. 20 donor maps. Initial cue retained.',
           '', '|w|Oracle intact|Fit intact|Oracle donor|Fit donor|Oracle delta|Fit delta|','|---|---:|---:|---:|---:|---:|---:|']
    for w in weights:
        s=summary[str(w)];keys=['oracle_intact_nll','fit_intact_nll','oracle_donor_nll','fit_donor_nll','oracle_delta_nll','fit_delta_nll']
        lines.append('|'+str(w)+'|'+'|'.join(f"{s[k]['mean']:.4f}" for k in keys)+'|')
    lines+=['','This is a small known-family reference pilot, not proof of parameter or neural mechanism recovery.',
            'Absolute delta NLL is primary; relative deltas, expected KL and probability-response disagreement are in summary.json.',
            'The donor input can be misleading, so its NLL may exceed random. This is not a choice-only optimal baseline.',
            'No neural training, test evaluation, or model selection on the evaluation maps.']
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
