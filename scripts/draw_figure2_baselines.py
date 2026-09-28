"""Figure 2 companion: frozen train-fitted controls, no fitting/test access."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import draw_iclr_main_drafts as draft

ROOT=Path('outputs/task1_with_llama_20260917')
BASE=Path('outputs/task1_diagnostics_20260916/baseline_fits.json')
OUT=Path('outputs/iclr_main_figures_draft_20260921')

def main():
    s=json.loads((ROOT/'summary.json').read_text())
    main=json.loads((draft.T1/'A_summary.json').read_text())
    fits=json.loads(BASE.read_text())
    for w in draft.W:
        for m in ('gru','transformer','llama'):
            np.testing.assert_allclose(s[str(float(w))]['all'][m+'_choice_nll']['mean'],main[str(float(w))]['all'][m+'_choice_nll']['mean'],atol=1e-12)
    draft.style()
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.8),sharey=True)
    names=[('static_arm_frequency','Training-set choice frequencies'),('static_stay_probability','Training-set stay probability'),
           ('static_transition','First-order transition model'),('train_selected_lag','Training-selected lag model')]
    for ax,(key,title),letter in zip(axs.flat,names,'abcd'):
        for m in ('gru','transformer','llama'):
            draft.curve(ax,[s[str(float(w))]['all'][m+'_choice_nll'] for w in draft.W],m)
        rows=[s[str(float(w))]['all']['baseline_'+key+'_nll'] for w in draft.W]
        ci=np.array([x['ci95'] for x in rows])
        ax.fill_between(draft.W,ci[:,0],ci[:,1],color='#b87826',alpha=.12,lw=0)
        ax.plot(draft.W,[x['mean'] for x in rows],color='#b87826',ls='--',marker='s',ms=3.2,label='Behavioral baseline')
        ax.axhline(np.log(4),color='#999',ls=':',lw=1,label='Uniform random')
        draft.panel(ax,letter,title);ax.set_ylim(.28,1.48)
    for ax in axs[:,0]:ax.set_ylabel('Validation choice NLL (nats)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.53,1),columnspacing=1.2,handlelength=1.7)
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.83,hspace=.55,wspace=.2)
    draft.save(fig,'FigureS1_choice_only_baselines',list(axs.flat))
    lines=['# Figure S1: behavioral baselines accompanying Figure 2','',
      'Choice-only GRU, Transformer and fine-tuned LLaMA compared with four training-fitted, reward-free behavioral baselines in the restless task. Each panel repeats the same neural curves to avoid a crowded combined legend. Orange denotes the baseline named in that panel; gray dotted line is uniform random (NLL = log 4). Oracle is omitted here because it observes rewards.',
      '', '## Baselines',
      '- Choice frequencies: pooled training-set proportions of A/B/C/D, fixed during evaluation.',
      '- Stay probability: pooled training-set repeat probability, assigned to the immediately previous choice; remaining mass divided equally among other arms.',
      '- First-order transition model: training-fitted 4 by 4 transition matrix with unit Laplace pseudocount, conditioned on the actual previous choice.',
      '- Lag model: lag selected from 1 through 20 using training NLL; repeat probability also fitted on training. Predictions use only the actual choice at the selected past lag. No validation tuning or online parameter updates.',
      '', '## Evaluation and uncertainty',
      'Exact same curves as Figure 2 right panel: 250 validation participants per weight, trials 151-200. GRU/Transformer metrics averaged across three seeds; LLaMA one seed. Bands are participant-bootstrap 95% intervals conditional on fitted models. Validation selected neural checkpoints. No sealed-test results are shown.',
      '', '## Selected lags',
      ', '.join(f"w={w}: lag {v['selected']['lag']}" for w,v in fits.items()),
      '', '## Interpretation',
      'Compare against tested baseline families only; predictive superiority does not identify hidden reward recovery or rule out more complex behavioral regularities. Overlapping stay/lag curves are expected when the selected lag is one.']
    violations=[]
    for w,v in s.items():
        for m in ('gru','transformer','llama'):
            for key,_ in names:
                if v['all'][m+'_choice_nll']['mean']>=v['all']['baseline_'+key+'_nll']['mean']:violations.append([w,m,key])
    lines += ['', 'Mean-NLL comparisons not favoring the neural model: '+json.dumps(violations)+'. This is a descriptive comparison, not a multiplicity-adjusted significance claim.']
    (OUT/'FigureS1_caption.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'FigureS1_provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'summary.json',BASE,draft.T1/'A_summary.json']},indent=2))
    print('Created Figure S1; mean comparison exceptions:',violations)

if __name__=='__main__':main()
