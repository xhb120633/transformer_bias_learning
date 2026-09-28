"""Audit and plot the seven-weight fixed-model operation sweep (validation only)."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from run_reference_followup import load, WEIGHTS
from paper_plot_style import weight_axis


T = Path('outputs/weight_operation_sweep_transformer_20260923')
L = Path('outputs/weight_llama_operation_fixedshape_20260924/production')
MAIN = Path('outputs/task1_main_integrated_20260918')
OUT = Path('outputs/weight_operation_sweep_figure_s9_fixedshape_20260924')
OPS = ('donor', 'neutral50', 'placeholder', 'delete')
LLAMA_OPS = OPS + ('history_only',)
COLORS = {'donor':'#475569','neutral50':'#b87826','placeholder':'#7956a6',
          'delete':'#26866b','history_only':'#c65e46'}


def choice_prob(logp):
    v = logp.astype(float)
    assert np.isfinite(v).all()
    p = np.exp(v-logsumexp(v,axis=-1,keepdims=True))
    np.testing.assert_allclose(p.sum(-1), 1., atol=1e-12)
    assert (p > 0).all()
    return p


def nll(p, actions):
    return -np.log(np.take_along_axis(p,actions[...,None],axis=-1)[...,0])


def main():
    assert (T/'COMPLETE.json').exists()
    audit=json.loads((L/'DOWNLOAD_AUDIT.json').read_text())
    assert audit['participant_files'] == 1750
    assert audit['fixed_shape'] and audit['job_array'] == '13530770'
    OUT.mkdir(exist_ok=True, parents=True)
    summary = {}
    max_first_error = 0.
    max_future_error = 0.
    for w in WEIGHTS:
        code = f'{round(w*100):03d}'
        d = load(w)
        actions = d['action'][:,150:]
        trows=[]
        for seed in (11,22,33):
            with np.load(T/f'transformer_reward_w{code}_full_seed{seed}.npz') as z:
                np.testing.assert_array_equal(z['participants'],d['base_participant_id'])
                trows.append({op:z[('zero_embedding' if op=='placeholder' else op)+'_delta_nll'].astype(float)
                              for op in OPS})
        rows=[]
        for i in range(250):
            with np.load(L/f'reward_w{code}'/f'participant_{i:03d}.npz') as z:
                row = dict(z)
            assert int(row['index']) == i
            assert str(row['participant']) == str(d['base_participant_id'][i])
            np.testing.assert_array_equal(row['actions'],d['action'][i])
            for op in LLAMA_OPS[1:]:
                assert row[op+'_logp'].shape == (200,4)
            errors=json.loads(str(row['first_choice_errors']))
            assert int(row['fixed_length']) == 2295
            assert int(row['delete_length']) == int(row['fixed_length']) - 200
            assert int(row['history_only_length']) < int(row['fixed_length'])
            for op in ('neutral50','placeholder','delete'):
                observed=float(np.max(np.abs(row[op+'_logp'][0]-row['intact_logp'][0])))
                np.testing.assert_allclose(observed,errors[op+'_logp'],atol=1e-12)
                max_first_error=max(max_first_error,observed)
            if i == 0:
                future=float(row['future_error'])
                assert np.isfinite(future)
                max_future_error=max(max_future_error,future)
            rows.append(row)
        intact=choice_prob(np.stack([r['intact_logp'][150:] for r in rows]))
        baseline=nll(intact,actions).mean(1)
        with np.load(MAIN/f'reward_w{code}_llama.npz') as z:
            np.testing.assert_array_equal(z['participants'],d['base_participant_id'])
            np.testing.assert_allclose(z['intact'],intact,atol=1e-7)
            donor_delta=z['delta_nll'].astype(float)
        lvals={'donor':donor_delta}
        for op in LLAMA_OPS[1:]:
            p=choice_prob(np.stack([r[op+'_logp'][150:] for r in rows]))
            lvals[op]=nll(p,actions).mean(1)-baseline
        summary[str(w)]={
            'Transformer':{op:float(np.mean([r[op].mean() for r in trows])) for op in OPS},
            'LLaMA SFT':{op:float(lvals[op].mean()) for op in LLAMA_OPS}}
        np.savez_compressed(OUT/f'reward_w{code}_vectors.npz',
            participants=d['base_participant_id'],
            **{f'transformer_seed{seed}_{op}':trows[j][op] for j,seed in enumerate((11,22,33)) for op in OPS},
            **{f'llama_{op}':lvals[op] for op in LLAMA_OPS})
    # Reward edits preserve the first causal prefix; the history-only rewrite
    # changes the instruction, so it is excluded from this invariance test.
    assert max_first_error < .003, f'First-choice numerical invariance failed: {max_first_error}'
    assert max_future_error < .003, f'Future-token invariance failed: {max_future_error}'
    (OUT/'summary.json').write_text(json.dumps({'split':'val','weights':list(WEIGHTS),
        'participants_per_weight':250,'scored_trials':'151-200','llama_seeds':1,
        'transformer_seeds':3,'max_first_choice_logp_error':max_first_error,
        'max_future_choice_logp_error':max_future_error,'fixed_inference_length':2295,
        'definition':{'donor':'matched other-participant reward history',
                      'neutral50':'all reward numbers replaced by 50',
                      'placeholder':'Transformer zero reward embeddings; LLaMA reserved special token',
                      'delete':'remove reward tokens and shift subsequent positions',
                      'history_only':'LLaMA-only rewrite of instruction and trial text to choice history'},
        'means':summary},indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.6),sharey=True)
    labels={'donor':'Donor rewards','neutral50':'Constant 50',
            'placeholder':'Reward placeholder','delete':'Delete reward tokens',
            'history_only':'Choice-history rewrite'}
    for ax,letter,model in zip(axs,'ab',('Transformer','LLaMA SFT')):
        for op in (OPS if model=='Transformer' else LLAMA_OPS):
            ax.plot(WEIGHTS,[summary[str(w)][model][op] for w in WEIGHTS],'-o',
                    color=COLORS[op],markersize=3,label=labels[op])
        ax.axhline(0,color='#999',lw=.6)
        ax.set_title(letter+'  '+model,loc='left',fontsize=9)
        weight_axis(ax)
    axs[0].set_ylabel(r'$\Delta$NLL (nats / choice)')
    fig.legend(*axs[1].get_legend_handles_labels(),frameon=False,fontsize=6.7,
               ncol=3,loc='upper center',bbox_to_anchor=(.54,1.0),columnspacing=1.6)
    fig.subplots_adjust(left=.085,right=.99,bottom=.17,top=.78,wspace=.16)
    for ext in ('pdf','png','svg'):
        fig.savefig(OUT/f'FigureS9_operation_sweep.{ext}',dpi=240)
    plt.close(fig)
    sources=[T/'COMPLETE.json',L/'DOWNLOAD_AUDIT.json',Path(__file__)]
    (OUT/'COMPLETE.json').write_text(json.dumps({
        'scientific_audit_passed':True,'split':'val','test_used':False,
        'participant_exports':1750,'weights':7,
        'max_first_choice_logp_error':max_first_error,
        'max_future_choice_logp_error':max_future_error,
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    },indent=2))
    print('COMPLETE',OUT.resolve(),'max_first_choice_error',max_first_error)


if __name__=='__main__':
    main()
