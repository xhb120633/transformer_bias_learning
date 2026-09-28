"""Audit and merge validation-only spatial LLaMA exports with existing A/B."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from ablate_spatial_references import summarize, nll
from run_rw_baseline import metrics

WEIGHTS=[0.,.1,.3,.5,.7,.9,1.]
INPUT=Path('outputs/spatial_llama_eval_v1/production')
REF=Path('outputs/spatial_neural_analysis_v1')
OUT=Path('outputs/spatial_main_integrated_20260918')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def probability(lp): return np.exp(lp.astype(float)-logsumexp(lp.astype(float),axis=-1,keepdims=True))

def main():
    global INPUT,OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=INPUT)
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();INPUT=args.input;OUT=args.output
    fixed='fixedshape_v2' in str(INPUT)
    required=[INPUT/f'reward_w{round(w*100):03d}'/mode/f'participant_{i:03d}.npz'
              for w in WEIGHTS for mode in ('full','choice_only') for i in range(50)]
    assert all(p.is_file() for p in required), 'Incomplete exports'
    a=json.loads((REF/'summary.json').read_text())
    b=json.loads(Path('outputs/rw_pooled_baseline_v1/summary.json').read_text())['spatial']
    OUT.mkdir(exist_ok=True)
    audit={}; lines=['# Spatial task: integrated LLaMA results','',
      'Validation only. A: eight maps, 160 choices/person. B: maps 7-8, 40 choices/person; 20 matched donor permutations. 50 participants/weight.',
      'LLaMA seed100; GRU/Transformer three seeds. All families use valid-choice conditional probabilities. Raw token NLL and valid-choice mass retained separately.',
      'Actual choice history and initial cue retained during B interventions. Each map is an independent sequence. Bootstrap intervals condition on fitted models and donor maps; validation also used for checkpoint selection. No sealed test used.',
      'Generator refits use within-participant calibration; RW and neural fits are pooled across training participants.','',
      '|Weight|A full NLL|A choice-only NLL|B intact NLL|B donor NLL|B delta NLL|B response error|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}'
        z=dict(np.load(REF/(name+'_matched.npz')))
        actions=z['actions']; maps=z['donor_maps']; ids=z['participants']
        payload=Path('data/spatial_llama_eval_v1')/(name+'.json')
        data=json.loads(payload.read_text())
        np.testing.assert_array_equal(actions,data['actions']);np.testing.assert_array_equal(maps,data['donor_maps'])
        np.testing.assert_array_equal(ids.astype(str),[str(r['participant']) for r in data['records']])
        exports={}; manifests={}
        for mode in ('full','choice_only'):
            folder=INPUT/name/mode; manifest=json.loads((folder/'COMPLETE.json').read_text())
            assert manifest['complete'] and not manifest['smoke'] and not manifest['test_used']
            assert manifest['weight']==name[-3:] and manifest['mode']==mode
            assert manifest['source_sha256']==sha(payload)
            # Original v1 evaluator hash, audited before the fixed-shape repair.
            expected_hash=sha(Path('scripts/eval_spatial_llama.py')) if fixed else 'c30e54ba4a550b9d52c5f86c0404e47ea4c2f477215813b0c564b5fc5f147eac'
            assert manifest['script_sha256']==expected_hash
            if fixed: assert manifest['fixed_length']==1024
            assert manifest['causal_max_difference'] < .003
            rows=[]
            for i in range(50):
                r=dict(np.load(folder/f'participant_{i:03d}.npz'))
                assert int(r['index'])==i and str(r['participant'])==str(ids[i])
                np.testing.assert_array_equal(r['actions'],actions[i]);np.testing.assert_array_equal(r['donor_indices'],maps[:,i])
                assert r['intact_logp'].shape==(8,20,25)
                if mode=='full': assert r['donor_logp'].shape==(20,2,20,25)
                for k,v in r.items():
                    if k.endswith('logp'): assert np.isfinite(v).all() and np.all(np.exp(v).sum(-1)<=1.0001)
                rows.append(r)
            exports[mode]={k:np.stack([r[k] for r in rows],axis=1 if k=='donor_logp' else 0) for k in rows[0] if k.endswith('logp')}
            manifests[mode]=manifest
        f=exports['full'];c=exports['choice_only']
        p=probability(f['intact_logp']);cp=probability(c['intact_logp']);q=probability(f['donor_logp'])
        op=z['oracle_B_intact'].reshape(50,40,25);oq=z['oracle_B_donor'].reshape(20,50,40,25)
        acts=actions[:,6:].reshape(50,40)
        v=metrics(p[:,6:].reshape(50,40,25),q.reshape(20,50,40,25),op,oq,acts)
        for mode,prob,lp in [('full',p,f['intact_logp']),('choice_only',cp,c['intact_logp'])]:
            v[mode+'_A_nll']=nll(prob,actions).mean((1,2))
            v[mode+'_A_accuracy']=(prob.argmax(-1)==actions).mean((1,2))
            v[mode+'_A_token_nll']=-np.take_along_axis(lp,actions[...,None],-1)[...,0].mean((1,2))
            v[mode+'_A_valid_mass']=np.exp(lp).sum(-1).mean((1,2))
            a[str(w)]['llama_'+mode+'_A_nll']=summarize(v[mode+'_A_nll'])
            a[str(w)]['llama_'+mode+'_A_accuracy']=summarize(v[mode+'_A_accuracy'])
        v['A_reward_predictive_gain']=v['choice_only_A_nll']-v['full_A_nll']
        firstdiff=np.abs(q[:,:,:,0]-p[None,:,6:,0])
        with np.load(Path('outputs/rw_pooled_baseline_v1')/f'spatial_{name}.npz') as rz:
            np.testing.assert_array_equal(rz['participants'],ids);np.testing.assert_array_equal(rz['donor_maps'],maps)
            for k,x in metrics(op,oq,op,oq,acts).items(): np.testing.assert_allclose(x,rz['oracle_'+k],atol=1e-10)
            v['nll_minus_rw']=v['intact_nll']-rz['rw_intact_nll']
            v['response_error_minus_rw']=v['response_error']-rz['rw_response_error']
        for k,x in v.items(): b[str(w)]['llama_'+k]=summarize(x)
        np.testing.assert_allclose(v['donor_nll']-v['intact_nll'],v['delta_nll'],atol=1e-12)
        audit[str(w)]=dict(manifests=manifests,identity_actions_maps_verified=True,oracle_matches_existing=True,
            first_choice_probability_max_difference=float(firstdiff.max()),first_choice_probability_mean_difference=float(firstdiff.mean()),
            full_valid_mass=float(v['full_A_valid_mass'].mean()),choice_valid_mass=float(v['choice_only_A_valid_mass'].mean()))
        np.savez_compressed(OUT/(name+'_llama.npz'),participants=ids,actions=actions,donor_maps=maps,intact=p,choice=cp,donor=q,**v)
        line='|'+str(w)+'|'+'|'.join(f'{v[k].mean():.5f}' for k in ('full_A_nll','choice_only_A_nll','intact_nll','donor_nll','delta_nll','response_error'))+'|'
        lines.append(line);print(line,flush=True)
    (OUT/'A_summary.json').write_text(json.dumps(a,indent=2));(OUT/'B_summary.json').write_text(json.dumps(b,indent=2))
    (OUT/'audit.json').write_text(json.dumps(dict(weights=audit,input_sha256={str(p):sha(p) for p in required}),indent=2))
    first_max=max(v['first_choice_probability_max_difference'] for v in audit.values())
    if first_max>=.001: lines += ['', '## Audit warning',
        'PROVISIONAL ONLY. Actual intact/donor first-choice probabilities differ despite identical causal prefixes (maximum absolute difference about .072). Variable batch padding is a suspected numerical confound, not a demonstrated cause. Same-shape smoke checks alone did not detect it. A fixed-shape reevaluation with actual donor-prefix checks is required before using these LLaMA ablation results in the paper.',
        'The pre-existing GRU/Transformer and symbolic results are unchanged. Preliminary LLaMA comparisons must not be described as audited replication.']
    (OUT/'report.md').write_text('\n'.join(lines))
    plot(a,b,provisional=first_max>=.001)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(aggregation_complete=True,scientific_audit_passed=first_max<.001,
        status='provisional_pending_fixedshape_evaluation' if first_max>=.001 else 'audited',
        input_files=700,test_used=False,llama_seeds=1,first_choice_probability_max_difference=first_max)))

def plot(a,b,provisional=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_plot_style import apply_style,weight_axis,COLORS,NLL_LABEL,WEIGHT_TICKS
    apply_style()
    labels={'oracle':'Oracle','fit':'Fitted generator','rw':'RW + softmax','gru':'GRU','transformer':'Transformer','llama':'LLaMA'}
    def curve(ax,rows,m):
        ci=np.array([x['ci95'] for x in rows])
        ax.plot(WEIGHTS,[x['mean'] for x in rows], '--' if m=='fit' else '-', marker='o',ms=4,lw=2,color=COLORS[m],label=labels[m])
        ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],color=COLORS[m],alpha=.1)
    def save(fig,name,axes):
        for ax in np.asarray(axes).flat: np.testing.assert_allclose(ax.get_xticks(),WEIGHT_TICKS)
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
    if provisional: fig.suptitle('PROVISIONAL: fixed-shape evaluation audit pending',color='#a33',fontsize=10,y=.89)
    for ax,mode,title in zip(axes,('full','choice_only'),('Full-input training','Choice-only training')):
        for m in ('gru','transformer','llama'):curve(ax,[a[str(w)][m+'_'+mode+'_A_nll'] for w in WEIGHTS],m)
        curve(ax,[a[str(w)]['oracle_full_A_nll'] for w in WEIGHTS],'oracle')
        ax.axhline(np.log(25),ls=':',color='#aaa',label='Uniform random');ax.set(title=title,ylabel=NLL_LABEL);weight_axis(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False)
    fig.text(.06,.025,'Spatial task | All 8 validation maps; 50 participants/weight. GRU/Transformer: 3 seeds; LLaMA: 1 seed.\nBands: participant-bootstrap 95% CI conditional on fits. Oracle observes rewards in both panels. Validation used for early stopping.',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.91));save(fig,'A_main',axes)
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    if provisional: fig.suptitle('PROVISIONAL: fixed-shape evaluation audit pending',color='#a33',fontsize=10,y=.955)
    for ax,metric,title in zip(axes.flat,('intact_nll','donor_nll','delta_nll','response_error'),('Original rewards','Donor-replaced rewards','Ablation effect','Probability-response error vs. oracle')):
        for m in labels:curve(ax,[b[str(w)][m+'_'+metric] for w in WEIGHTS],m)
        ax.set(title=title,ylabel='Probability-response error' if metric=='response_error' else ('Delta NLL (nats)' if metric=='delta_nll' else NLL_LABEL));weight_axis(ax)
        if metric in ('intact_nll','donor_nll'):ax.axhline(np.log(25),color='#aaa',ls=':')
        else:ax.axhline(0,color='#aaa',lw=.7)
    lo=min(ax.get_ylim()[0] for ax in axes[0]);hi=max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:ax.set_ylim(lo,hi)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=6,frameon=False,fontsize=10)
    fig.text(.06,.025,'Spatial task | Validation maps 7-8; 50 participants/weight; 20 matched donor maps. Choices and initial cue fixed.\nDelta NLL = donor minus original. Bands condition on fits/maps. GRU/Transformer: 3 seeds; LLaMA: 1 seed.\nGenerator refits use within-participant calibration. Development validation, not independent test confirmation.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));save(fig,'B_main',axes)

if __name__=='__main__':main()
