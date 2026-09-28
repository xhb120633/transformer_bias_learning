"""Complete seven-weight operation display with neural and symbolic references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_reference_followup import WEIGHTS, load, params
from run_weight_curve_a import replay, nll
from paper_plot_style import weight_axis


BASE = Path('outputs/weight_operation_sweep_figure_s9_fixedshape_20260924')
GRU = Path('outputs/weight_operation_sweep_gru_20260924')
REFERENCE = Path('outputs/weight_generator_fit_comparison_20260916/participant_vectors.npz')
FITS = Path('outputs/reference_followup_fit_fine_20260916')
PROTOCOL = Path('outputs/weight_curve_a_20260915/protocol.json')
OUT = Path('outputs/weight_operation_sweep_complete_20260924')
OPS = ('donor', 'neutral50', 'placeholder', 'delete')
NAMES = {'donor':'Donor rewards', 'neutral50':'Constant 50',
         'placeholder':'Reward placeholder', 'delete':'Delete reward tokens',
         'history_only':'Choice-history rewrite'}
COLORS = {'donor':'#475569', 'neutral50':'#b87826', 'placeholder':'#7956a6',
          'delete':'#26866b', 'history_only':'#c65e46'}


def main():
    assert json.loads((BASE/'COMPLETE.json').read_text())['scientific_audit_passed']
    ga = json.loads((GRU/'COMPLETE.json').read_text())
    assert ga['runs'] == 21 and ga['max_first_choice_error'] < .003
    source = json.loads((BASE/'summary.json').read_text())
    scale = json.loads(PROTOCOL.read_text())['q_scale']
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with np.load(REFERENCE) as reference:
        for w in WEIGHTS:
            code = f'{round(w*100):03d}'
            data = load(w)
            ids = data['base_participant_id']
            src = source['means'][str(w)]
            gru_rows = []
            for seed in (11,22,33):
                with np.load(GRU/f'gru_reward_w{code}_full_seed{seed}.npz') as z:
                    np.testing.assert_array_equal(z['participants'], ids)
                    gru_rows.append({op:z[('zero_embedding' if op=='placeholder' else op)+'_delta_nll'].astype(float)
                                     for op in OPS})
            rd = {op: float(np.mean([row[op].mean() for row in gru_rows])) for op in OPS}
            with np.load(FITS/f'reward_w{code}_fits.npz') as f:
                fit_parameters = f['parameters'][:,0].astype(float)
            assert fit_parameters.shape == (250,3)
            fitted = params(data, fit_parameters[:,0], fit_parameters[:,1], fit_parameters[:,2])
            actual_reward = data['reward']
            constant_reward = np.full_like(actual_reward, 50)
            oracle_intact = replay(data, actual_reward, scale)
            np.testing.assert_allclose(oracle_intact, data['choice_probability'], atol=1e-7)
            fit_intact = replay(fitted, actual_reward, scale)
            for name, model, intact in [('oracle',data,oracle_intact),('fit',fitted,fit_intact)]:
                changed = replay(model, constant_reward, scale)
                np.testing.assert_allclose(changed[:,0], intact[:,0], atol=1e-12)
                base = nll(intact,data['action'])[:,150:].mean(1)
                delta = nll(changed,data['action'])[:,150:].mean(1)-base
                np.testing.assert_allclose(base,reference[f'{w}__{name}_intact_nll'],atol=1e-8)
                summary.setdefault(str(w),{})['Oracle' if name=='oracle' else 'Fitted generator'] = {
                    'donor':float(reference[f'{w}__{name}_donor'].mean()),
                    'neutral50':float(delta.mean())}
                np.savez_compressed(OUT/f'reward_w{code}_{name}_constant.npz',
                                    participants=ids,constant50_delta_nll=delta,
                                    donor_delta_nll=reference[f'{w}__{name}_donor'])
            summary[str(w)]['GRU']=rd
            summary[str(w)]['Transformer']=src['Transformer']
            summary[str(w)]['LLaMA SFT']=src['LLaMA SFT']
            print('COMPLETE',w,{k:round(v['neutral50'],4) for k,v in summary[str(w)].items()
                                if 'neutral50' in v},flush=True)

    (OUT/'summary.json').write_text(json.dumps({'split':'val','weights':list(WEIGHTS),
        'participants_per_weight':250,'score_trials':'151-200',
        'GRU_seeds':3,'Transformer_seeds':3,'LLaMA_SFT_seeds':1,
        'symbolic_reference':'true parameters or individual prefix-fitted generator; numeric operations only',
        'means':summary},indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig = plt.figure(figsize=(7.1,5.7))
    grid = fig.add_gridspec(2,6,height_ratios=[1,1],left=.085,right=.99,
                            bottom=.105,top=.87,hspace=.57,wspace=.7)
    axes = [fig.add_subplot(grid[0,0:2]),fig.add_subplot(grid[0,2:4]),
            fig.add_subplot(grid[0,4:6]),fig.add_subplot(grid[1,0:3]),
            fig.add_subplot(grid[1,3:6])]
    names = ('GRU','Transformer','LLaMA SFT','Oracle','Fitted generator')
    for idx,(ax,name) in enumerate(zip(axes,names)):
        ops = OPS + (('history_only',) if name=='LLaMA SFT' else ()) if idx<3 else ('donor','neutral50')
        for op in ops:
            ax.plot(WEIGHTS,[summary[str(w)][name][op] for w in WEIGHTS],'-o',
                    color=COLORS[op],markersize=2.7,lw=1.5,label=NAMES[op])
        ax.axhline(0,color='#999',lw=.6)
        ax.set_title('abcde'[idx]+'  '+name,loc='left',fontsize=9)
        weight_axis(ax)
        ax.set_xlabel('Generating reward weight',fontsize=7)
        if idx in (0,3): ax.set_ylabel(r'$\Delta$NLL (nats / choice)',fontsize=8)
        ax.set_ylim((-.04,.76) if idx<3 else (-.12,2.22))
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7,ncol=3,
               loc='upper center',bbox_to_anchor=(.54,.985),columnspacing=1.7)
    for ext in ('pdf','png','svg'):
        fig.savefig(OUT/f'FigureS9_operation_sweep.{ext}',dpi=240)
    plt.close(fig)
    sources=[BASE/'COMPLETE.json',GRU/'COMPLETE.json',REFERENCE,PROTOCOL,Path(__file__)]
    sources += [FITS/f'reward_w{round(w*100):03d}_fits.npz' for w in WEIGHTS]
    (OUT/'COMPLETE.json').write_text(json.dumps({'scientific_audit_passed':True,
        'split':'val','test_used':False,'weights':7,'neural_families':3,
        'symbolic_references':2,'first_choice_max_gru':ga['max_first_choice_error'],
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))


if __name__=='__main__': main()
