"""Replot existing matched RW assays; no fitting or new data generation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_plot_style import (apply_style, weight_axis, COLORS, NLL_LABEL,
                              DELTA_LABEL, LINE, BAND_ALPHA, EXPORT_DPI)

ROOT=Path('outputs/rw_pooled_baseline_v1')
WEIGHTS=[0.,.1,.3,.5,.7,.9,1.]
STYLES=[('oracle','Oracle','#444444'),('fit','Fitted generator','#b87826'),
        ('rw','RW + softmax','#b14375'),('gru','GRU','#228b78'),
        ('transformer','Transformer','#536eba')]

def main():
    apply_style()
    summary=json.loads((ROOT/'summary.json').read_text())
    for task in ('restless','spatial'):
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
        for ax,metric,title in zip(axes,['intact_nll','donor_nll'],['Original rewards','Reward ablation (donor replacement)']):
            for key,label,color in STYLES:
                rows=[summary[task][str(w)][key+'_'+metric] for w in WEIGHTS]
                means=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows])
                ax.plot(WEIGHTS,means,color=COLORS[key],label=label,**LINE)
                ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],color=COLORS[key],alpha=BAND_ALPHA)
            ax.axhline(np.log(4 if task=='restless' else 25),color='#999',ls=':',label='Uniform random')
            ax.set(title=title,ylabel=NLL_LABEL)
            weight_axis(ax)
            ax.spines[['top','right']].set_visible(False)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        window='trials 151–200' if task=='restless' else 'maps 7–8'
        fig.text(.5,.02,f'Same validation targets ({window}); same fitted models before/after ablation. Not choice-only retraining.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.84))
        fig.savefig(ROOT/f'{task}_prediction_ablation.png',dpi=EXPORT_DPI)
        fig.savefig(ROOT/f'{task}_prediction_ablation.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        for key,label,color in STYLES:
            rows=[summary[task][str(w)][key+'_delta_nll'] for w in WEIGHTS]
            means=[r['mean'] for r in rows];ci=np.array([r['ci95'] for r in rows])
            ax.plot(WEIGHTS,means,'o-',color=color,label=label,lw=2,ms=4)
            ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],color=color,alpha=.1)
        ax.axhline(0,color='#999',lw=.8)
        ax.set(ylabel=DELTA_LABEL)
        weight_axis(ax)
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9)
        fig.tight_layout();fig.savefig(ROOT/f'{task}_delta_nll.png',dpi=180)
        fig.savefig(ROOT/f'{task}_delta_nll.pdf');plt.close(fig)

if __name__=='__main__':main()
