"""Rebuild frozen independent-test figures without training or model inference."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import draw_iclr_main_drafts as d
import draw_figure1_revision17 as overview
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('reproduced_figures'));args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];archive=root/'archive/outputs'
    d.OUT=args.output.resolve();d.OUT.mkdir(parents=True,exist_ok=True)
    d.T1=archive/'test_main_integrated_20260925/restless';d.T2=archive/'test_main_integrated_20260925/spatial';d.SPLIT_LABEL='Test'
    d.style();d.figure2();d.figure_b(d.T1,3);d.figure_b(d.T2,4)
    overview.REST=archive/'weight_curve_a_20260915/reward_w050/val_000.npz'
    overview.SPAT=archive/'spatial_pilot_v1/reward_w050/val_audit.npz'
    overview.EXAMPLE=archive/'figure1_publication_20260923/example_predictions.json'
    overview.SUMMARY=d.T1/'B_summary.json';overview.OUT=d.OUT;overview.main()
    a=json.loads((d.T1/'A_summary.json').read_text());d.style()
    names=[('static_arm_frequency','Training-set choice frequencies'),('static_stay_probability','Training-set stay probability'),('static_transition','First-order transition model'),('train_selected_lag','Training-selected lag model')]
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.8),sharey=True)
    for ax,(key,title),letter in zip(axs.flat,names,'abcd'):
        for m in ('gru','transformer','llama'):d.curve(ax,[a[str(float(w))]['all'][m+'_choice_nll'] for w in d.W],m)
        rows=[a[str(float(w))]['all']['baseline_'+key+'_nll'] for w in d.W];ci=np.array([x['ci95'] for x in rows])
        ax.fill_between(d.W,ci[:,0],ci[:,1],color='#b87826',alpha=.12,lw=0)
        ax.plot(d.W,[x['mean'] for x in rows],color='#b87826',ls='--',marker='s',ms=3.2,label='Behavioral baseline')
        ax.axhline(np.log(4),color='#999',ls=':',lw=1,label='Uniform random');d.panel(ax,letter,title);ax.set_ylim(.28,1.48)
    for ax in axs[:,0]:ax.set_ylabel('Test choice NLL (nats)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.53,1),columnspacing=1.2,handlelength=1.7)
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.83,hspace=.55,wspace=.2)
    d.save(fig,'FigureS1_choice_only_baselines',list(axs.flat))
    print('Rebuilt main test figures and behavioral-baseline supplement.')
if __name__=='__main__':main()
