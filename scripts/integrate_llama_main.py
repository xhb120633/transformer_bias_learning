"""Audit and integrate completed main assays independently of pending local assays."""
import hashlib
import json
from pathlib import Path
import numpy as np
from collect_weight_llama import read_mode, probabilities
from run_reference_followup import ROOT, load, WEIGHTS
from run_weight_curve_a import replay
from run_rw_baseline import metrics
from ablate_spatial_references import summarize, nll

OUT=Path('outputs/task1_main_integrated_20260918')
INPUT=Path('outputs/weight_llama_eval_uncached_20260917')

def main():
    required=[INPUT/'production'/f'reward_w{round(w*100):03d}'/m/f'participant_{i:03d}.npz'
              for w in WEIGHTS for m in ('full','choice') for i in range(250)]
    assert all(p.exists() for p in required)
    OUT.mkdir(exist_ok=True)
    a=json.loads(Path('outputs/task1_diagnostics_20260916/summary.json').read_text())
    b=json.loads(Path('outputs/rw_pooled_baseline_v1/summary.json').read_text())['restless']
    scale=json.loads((ROOT/'protocol.json').read_text())['q_scale']
    lines=['# Main task: completed LLaMA integration','',
      'Validation trials 151-200, 250 participants per weight, 20 matched donor maps. LLaMA seed100; GRU/Transformer metric averages over seeds11/22/33. No local assay required or imputed. No test accessed.',
      'All families use probabilities conditional on the four valid choices. Raw LLaMA token NLL and valid-choice mass are retained. CIs resample participants conditional on fitted models and donor maps. Validation also selected checkpoints.',
      'Generator refits use individual prefix calibration, unlike pooled RW and neural fitting.','',
      '|Weight|LLaMA full NLL|Choice-only NLL|Donor NLL|Delta NLL|Response error|',
      '|---|---:|---:|---:|---:|---:|']
    audits={}
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}';d=load(w);actions=d['action'][:,150:]
        maps=np.load(ROOT/name/'validation_assay.npz')['donor_mappings']
        f=read_mode(INPUT,name,'full',d,maps);c=read_mode(INPUT,name,'choice',d,maps)
        p=probabilities(f['intact_logp'][:,150:]);q=probabilities(f['donor_logp']);cp=probabilities(c['intact_logp'][:,150:])
        for arr in (p,q,cp):
            assert np.isfinite(arr).all();np.testing.assert_allclose(arr.sum(-1),1,atol=1e-7)
        op=replay(d,d['reward'],scale)[:,150:]
        oq=np.array([replay(d,d['reward'][m],scale)[:,150:] for m in maps])
        v=metrics(p,q,op,oq,actions)
        v['choice_nll']=nll(cp,actions).mean(1)
        v['reward_predictive_gain']=v['choice_nll']-v['intact_nll']
        v['full_vocabulary_nll']=-np.take_along_axis(f['intact_logp'][:,150:],actions[...,None],-1)[...,0].mean(1)
        v['choice_full_vocabulary_nll']=-np.take_along_axis(c['intact_logp'][:,150:],actions[...,None],-1)[...,0].mean(1)
        v['valid_choice_mass']=np.exp(f['intact_logp'][:,150:]).sum(-1).mean(1)
        with np.load(Path('outputs/rw_pooled_baseline_v1')/f'restless_{name}.npz') as z:
            np.testing.assert_array_equal(z['participants'],d['base_participant_id'])
            np.testing.assert_array_equal(z['donor_maps'],maps)
            ov=metrics(op,oq,op,oq,actions)
            for k,x in ov.items():np.testing.assert_allclose(x,z['oracle_'+k],atol=1e-9)
            v['nll_minus_rw']=v['intact_nll']-z['rw_intact_nll']
            v['response_error_minus_rw']=v['response_error']-z['rw_response_error']
        np.testing.assert_allclose(v['donor_nll']-v['intact_nll'],v['delta_nll'],atol=1e-12)
        for k,x in v.items():b[str(w)]['llama_'+k]=summarize(x)
        for k,source in [('full_nll','intact_nll'),('choice_nll','choice_nll'),('reward_predictive_gain','reward_predictive_gain')]:
            a[str(w)]['all']['llama_'+k]=summarize(v[source])
        np.savez_compressed(OUT/f'{name}_llama.npz',participants=d['base_participant_id'],donor_maps=maps,intact=p,choice=cp,donor=q,**v)
        audits[str(w)]={'participants':250,'donors':20,'identity_actions_maps_verified':True,'oracle_replay_matches_existing':True,'valid_choice_mass_mean':float(v['valid_choice_mass'].mean())}
        vals=[v[k].mean() for k in ('intact_nll','choice_nll','donor_nll','delta_nll','response_error')]
        lines.append('|'+str(w)+'|'+'|'.join(f'{x:.5f}' for x in vals)+'|')
        print(lines[-1],flush=True)
    (OUT/'A_summary.json').write_text(json.dumps(a,indent=2))
    (OUT/'B_summary.json').write_text(json.dumps(b,indent=2))
    (OUT/'audit.json').write_text(json.dumps({'weights':audits,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in required}},indent=2))
    lines.extend(['','## Interpretation',
      'LLaMA full-input mean NLL is below GRU and Transformer at every nonzero weight. Choice-only LLaMA is close to GRU; this does not identify which latent information either network recovered.',
      'At weights .7/.9/1, paired LLaMA-minus-RW contrasts show lower intact NLL and larger probability-response error; the participant-bootstrap intervals exclude zero in both directions at these weights. These are exploratory, unadjusted comparisons conditional on one LLaMA training seed.',
      'At weight1, LLaMA intact NLL=.38777, choice-only=.47622, donor NLL=.59375, delta=.20598, response error=.53192. A positive retraining benefit (.08845 nats) coexists with a donor effect far below the oracle (.2 vs about2.06 nats). These are different estimands, not an information decomposition.',
      'Lower-weight comparisons do not show a uniform RW advantage. No universal prediction-fidelity tradeoff or complete mechanism recovery claim follows.'])
    (OUT/'report.md').write_text('\n'.join(lines))
    plot(a,b)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(main_complete=True,local_included=False,test_used=False,input_files=3500)))

def plot(a,b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_plot_style import apply_style,weight_axis,COLORS,NLL_LABEL
    apply_style()
    # RW and LLaMA must remain visually distinct when both appear.
    colors=dict(COLORS)
    labels={'oracle':'Oracle','fit':'Fitted generator','rw':'RW + softmax','gru':'GRU','transformer':'Transformer','llama':'LLaMA'}
    def curve(ax,rows,model):
        y=[r['mean'] for r in rows];ci=np.array([r['ci95'] for r in rows])
        ax.plot(WEIGHTS,y,'--' if model=='fit' else '-',marker='o',ms=4,lw=2,color=colors[model],label=labels[model])
        ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],color=colors[model],alpha=.10)
    def save(fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
    for ax,mode,title in zip(axes,('full','choice'),('Full-input training','Choice-only training')):
        for m in ('gru','transformer','llama'):curve(ax,[a[str(w)]['all'][m+'_'+mode+'_nll'] for w in WEIGHTS],m)
        curve(ax,[a[str(w)]['all']['oracle_nll'] for w in WEIGHTS],'oracle')
        ax.axhline(np.log(4),color='#aaa',ls=':',label='Uniform random')
        ax.set(title=title,ylabel=NLL_LABEL);weight_axis(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False)
    fig.text(.06,.025,'Validation trials 151–200; 250 participants/weight. GRU/Transformer: 3 seeds; LLaMA: 1 seed.\nBands: participant-bootstrap 95% CI conditional on fits. Oracle observes rewards in both panels. Validation used for early stopping.',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.91));save(fig,'A_main')
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    for ax,metric,title in zip(axes.flat,('intact_nll','donor_nll','delta_nll','response_error'),('Original rewards','Donor-replaced rewards','Ablation effect','Probability-response error vs. oracle')):
        for m in labels:curve(ax,[b[str(w)][m+'_'+metric] for w in WEIGHTS],m)
        ax.set(title=title,ylabel='Probability-response error' if metric=='response_error' else ('Delta NLL (nats)' if metric=='delta_nll' else NLL_LABEL));weight_axis(ax)
        if metric in ('intact_nll','donor_nll'):ax.axhline(np.log(4),color='#aaa',ls=':')
        else:ax.axhline(0,color='#aaa',lw=.7)
    lo=min(ax.get_ylim()[0] for ax in axes[0]);hi=max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:ax.set_ylim(lo,hi)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=6,frameon=False,fontsize=10)
    fig.text(.06,.025,'Same validation targets and 20 donor maps; actual choices fixed. Delta NLL = donor − original.\nGRU/Transformer: 3 seeds; LLaMA: 1 seed. Bands condition on fits/maps. Generator refits use individual prefix calibration.\nDevelopment validation, not independent test confirmation. Local ablation is not included.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94));save(fig,'B_main')

if __name__=='__main__':main()
