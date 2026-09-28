"""Audit matched Llama exports and merge A/B diagnostics; never reads test data."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from complete_task1_diagnostics import loss, calibrate, temper, response_metrics, summarize
from run_reference_followup import ROOT, load, WEIGHTS
from run_weight_curve_a import replay


def read_mode(root, name, mode, d, maps):
    folder = root / 'production' / name / mode
    rows = []
    for i in range(250):
        with np.load(folder / f'participant_{i:03d}.npz') as z:
            r = dict(z)
        assert int(r['index']) == i
        assert str(r['participant']) == str(d['base_participant_id'][i])
        assert np.array_equal(r['actions'], d['action'][i])
        assert np.array_equal(r['donor_indices'], maps[:, i])
        assert r['intact_logp'].shape == (200, 4)
        for k, p in r.items():
            if k.endswith('logp'):
                assert np.isfinite(p).all() and np.all(np.exp(p).sum(-1) <= 1.0001)
        if mode == 'full':
            assert r['donor_logp'].shape == r['suffix_donor_logp'].shape == (20, 50, 4)
        if mode == 'local':
            assert np.array_equal(r['target_indices'], np.arange(150, 200))
            assert r['local_logp'].shape == (20, 50, 4)
            assert r['local_intact_logp'].shape == (50, 4)
        rows.append(r)
    return {k: np.stack([r[k] for r in rows], axis=1 if k in ('donor_logp', 'suffix_donor_logp', 'local_logp') else 0)
            for k in rows[0] if k.endswith('logp')}


def probabilities(lp):
    return np.exp(lp.astype(float) - logsumexp(lp.astype(float), axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('outputs/weight_llama_eval_uncached_20260917'))
    ap.add_argument('--output', type=Path, default=Path('outputs/task1_with_llama_20260917'))
    args = ap.parse_args()
    # Fail before writing any combined result if even one required participant is missing.
    required = [args.input/'production'/f'reward_w{round(w*100):03d}'/mode/f'participant_{i:03d}.npz'
                for w in WEIGHTS for mode in ('full','choice','local') for i in range(250)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f'{len(missing)}/{len(required)} exports missing; first: {missing[0]}')
    args.output.mkdir(parents=True, exist_ok=True)
    old = Path('outputs/task1_diagnostics_20260916')
    summary = json.loads((old/'summary.json').read_text())
    bs = json.loads(Path('outputs/section_b_weight_curves_20260916/summary.json').read_text())
    temps = {}
    scale = json.loads((ROOT/'protocol.json').read_text())['q_scale']
    for w in WEIGHTS:
        name = f'reward_w{round(w*100):03d}'
        d = load(w); a = d['action'][:, 150:]
        maps = np.load(ROOT/name/'validation_assay.npz')['donor_mappings']
        full = read_mode(args.input,name,'full',d,maps)
        co = read_mode(args.input,name,'choice',d,maps)
        local = read_mode(args.input,name,'local',d,maps)
        assert np.allclose(full['intact_logp'],local['intact_logp'],atol=.003,rtol=0)
        p = probabilities(full['intact_logp'][:,150:]); cp = probabilities(co['intact_logp'][:,150:])
        prior = dict(np.load(old/(name+'_diagnostics.npz')))
        folds = prior['folds']
        calp, ts = calibrate(p,a,folds);temps[str(w)] = ts
        base = loss(p,a); calbase = loss(calp,a)
        rawbase = -np.take_along_axis(full['intact_logp'][:,150:],a[...,None],-1)[...,0]
        rawco = -np.take_along_axis(co['intact_logp'][:,150:],a[...,None],-1)[...,0]
        v = dict(full_nll=base.mean(1),choice_nll=loss(cp,a).mean(1),
                 full_accuracy=(p.argmax(-1)==a).mean(1),choice_accuracy=(cp.argmax(-1)==a).mean(1),
                 reward_predictive_gain=(loss(cp,a)-base).mean(1),calibrated_full_nll=calbase.mean(1),
                 full_vocabulary_nll=rawbase.mean(1),choice_full_vocabulary_nll=rawco.mean(1),
                 intact_abcd_mass=np.exp(full['intact_logp'][:,150:]).sum(-1).mean(1))
        op = replay(d,d['reward'],scale)[:,150:]
        local_p = probabilities(local['local_intact_logp'])
        v['local_prefix_intact_nll'] = loss(local_p,a).mean(1)
        v['local_prefix_minus_full_nll'] = (loss(local_p,a)-base).mean(1)
        for mode in ('donor','suffix_donor','local'):
            lp = (local if mode=='local' else full)[mode+'_logp']
            pair_p = local_p if mode == 'local' else p
            pair_base = loss(pair_p,a)
            pair_calp = np.empty_like(pair_p)
            for k,t in enumerate(ts): pair_calp[folds==k] = temper(pair_p[folds==k],t)
            pair_calbase = loss(pair_calp,a)
            pair_rawbase = (-np.take_along_axis(local['local_intact_logp'],a[...,None],-1)[...,0]
                            if mode == 'local' else rawbase)
            q = probabilities(lp); oq=[]
            for m in maps:
                r=d['reward'][m].copy()
                if mode=='suffix_donor': r[:,:150]=d['reward'][:,:150]
                oq.append(replay(d,r,scale,mode=='local')[:,150:])
            oq=np.stack(oq);cq=np.empty_like(q)
            for k,t in enumerate(ts):cq[:,folds==k]=temper(q[:,folds==k],t)
            for label,pp,qq,bl in [('raw',pair_p,q,pair_base),('calibrated',pair_calp,cq,pair_calbase)]:
                v[label+'_'+mode+'_delta_nll']=(loss(qq,a[None])-bl[None]).mean((0,2))
                for k,x in response_metrics(pp,qq,op,oq).items():v[label+'_'+mode+'_'+k]=x
            v[mode+'_full_vocabulary_delta_nll']=(-np.take_along_axis(lp,a[None,...,None],-1)[...,0]-pair_rawbase[None]).mean((0,2))
            v[mode+'_abcd_mass']=np.exp(lp).sum(-1).mean((0,2))
            v[mode+'_ablation_minus_retraining_gap']=v['raw_'+mode+'_delta_nll']-v['reward_predictive_gain']
        for k,x in prior.items():
            if k.startswith('baseline_') and k.endswith('_nll'):
                v['choice_minus_'+k]=(v['choice_nll']-x)
        for k,x in v.items(): prior['llama_'+k]=x
        vec={k:x for k,x in prior.items() if k not in ('participants','sign','folds')}
        summary[str(w)] = summarize(vec,d['kernel_sign'])
        bv=dict(intact_nll=v['full_nll'],**{m+'_delta_nll':v['raw_'+m+'_delta_nll'] for m in ('donor','local','suffix_donor')},
                **{m+'_response_error':v['raw_'+m+'_response_error'] for m in ('donor','local','suffix_donor')})
        bs[str(w)]['LLaMA']=summarize(bv,d['kernel_sign'])
        np.savez_compressed(args.output/(name+'_diagnostics.npz'),**prior)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'section_b_summary.json').write_text(json.dumps(bs,indent=2))
    (args.output/'llama_temperatures.json').write_text(json.dumps(temps,indent=2))
    import plot_section_a_clean as aplot
    aplot.OUT=args.output;aplot.main()
    plot_b(bs,args.output)
    lines=['# Matched Llama seven-weight diagnostics','',
           'Validation only, targets 151-200; 250 matched participants, exact 20 donor maps. Llama: seed100; GRU/Transformer: metric means across seeds11/22/33.',
           'Main comparisons use ABCD-conditional probabilities for all families. Full-vocabulary Llama NLL/effects and ABCD probability mass are retained separately, not silently mixed.',
           'Validation was used for checkpoint selection. Bootstrap conditions on training seeds and donor maps. No sealed test data accessed.',
           'Five-fold temperature calibration uses full-sequence intact choices of other participants only; this does not remove checkpoint-selection reuse. Local effects use matched-length uncached prefix intact/ablated predictions, with the same fitted temperatures. Prefix/full NLL differences are exported separately.', '',
           '|Weight|Full NLL|Choice-only NLL|Donor delta|Local delta|Suffix delta|', '|---|---:|---:|---:|---:|---:|']
    for w in WEIGHTS:
        v=summary[str(w)]['all']
        lines.append('|'+str(w)+'|'+'|'.join(f"{v['llama_'+k]['mean']:.5f}" for k in ('full_nll','choice_nll','raw_donor_delta_nll','raw_local_delta_nll','raw_suffix_donor_delta_nll'))+'|')
    (args.output/'report.md').write_text('\n'.join(lines))
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(complete=True,test_used=False,llama_seed_count=1)))


def plot_b(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_plot_style import COLORS, apply_style, weight_axis
    apply_style()
    models={k:COLORS[v] for k,v in [('Oracle','oracle'),('Generator fit','fit'),('GRU','gru'),('Transformer','transformer'),('LLaMA','llama')]}
    for metric in ('delta_nll','response_error'):
        fig,axs=plt.subplots(3,3,figsize=(14,10),sharex=True)
        for j,g in enumerate(('all','positive','negative')):
            for i,m in enumerate(('donor','local','suffix_donor')):
                ax=axs[i,j]
                for model,c in models.items():
                    vals=[summary[str(w)][model][g][m+'_'+metric] for w in WEIGHTS]
                    ci=np.array([v['ci95'] for v in vals])
                    ax.plot(WEIGHTS,[v['mean'] for v in vals],marker='o',ms=3,color=c,label=model,ls='--' if model=='Generator fit' else '-')
                    ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],alpha=.1,color=c)
                ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.15);weight_axis(ax)
                if j==0:ax.set_ylabel(m.replace('_',' ')+'\n'+metric.replace('_',' '))
                if i==0:ax.set_title(g+' participants')
                if i==2:ax.set_xlabel('Generating reward weight')
        for row in axs:
            bounds=(min(a.get_ylim()[0] for a in row),max(a.get_ylim()[1] for a in row))
            for ax in row:ax.set_ylim(bounds)
        h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=5,bbox_to_anchor=(.5,.96),frameon=False)
        fig.suptitle('B | Matched reward interventions',fontsize=17)
        fig.subplots_adjust(top=.87,bottom=.14,hspace=.25,wspace=.25)
        fig.text(.07,.035,'Validation trials 151–200 | ABCD-conditional probabilities | 250 participants/weight | 20 matched donor maps\nLLaMA: 1 seed; GRU/Transformer: 3 seeds. Participant bootstrap CIs conditional on seeds/maps. Validation used for early stopping.',fontsize=9)
        for ext in ('png','pdf','svg'):fig.savefig(out/f'section_b_{metric}.{ext}',dpi=180)
        plt.close(fig)


if __name__=='__main__':main()
