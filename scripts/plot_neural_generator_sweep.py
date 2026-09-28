"""A-style weight curves, plus matched B curves. No seed on the x axis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/neural_generator_sweep_v1')
WEIGHTS=[0.,.1,.3,.5,.7,.9,1.]
HEADS=[101,102,103]
COLORS={'oracle':'#444444','rw':'#b14375','gru':'#228b78','transformer':'#536eba'}

def plot():
    complete=all((ROOT/f'head{h}_w{round(w*100):03d}'/'summary.json').exists() for h in HEADS for w in WEIGHTS)
    if not complete:return
    def values(key,b=False):
        rows=[]
        for h in HEADS:
            row=[]
            for w in WEIGHTS:
                root=ROOT/f'head{h}_w{round(w*100):03d}'
                if b:
                    with np.load(root/'B_vectors.npz') as z:row.append(float(z[key].mean()))
                else:row.append(json.loads((root/'summary.json').read_text())[key])
            rows.append(row)
        return np.array(rows)
    def line(ax,y,label,color):
        for row in y:ax.plot(WEIGHTS,row,color=color,lw=.7,alpha=.25)
        ax.plot(WEIGHTS,y.mean(0),'o-',color=color,label=label,lw=2,ms=4)
    def finish(fig,axes,name):
        for ax in axes:
            ax.set_xlabel('Reward-visible policy mixture weight')
            ax.set_xticks(WEIGHTS);ax.spines[['top','right']].set_visible(False)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,fontsize=9)
        fig.text(.5,.015,'Thick lines: mean across 3 generator heads; faint lines: each head. One student seed per cell. Validation, not test.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.90))
        fig.savefig(ROOT/f'{name}.png',dpi=180);fig.savefig(ROOT/f'{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,mode,title in zip(axes,['full','choice_only'],['Full-input training','Choice-only training']):
        for model,label in [('gru','GRU'),('transformer','Transformer')]:
            line(ax,values(f'{model}_{mode}_A'),label,COLORS[model])
        line(ax,values('oracle_A'),'Full-information oracle',COLORS['oracle'])
        ax.axhline(np.log(4),color='#999',ls=':',label='Uniform random')
        ax.set(title=title,ylabel='Validation NLL (nats / choice)')
    finish(fig,axes,'A_prediction')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in zip(axes,['intact_nll','donor_nll','delta_nll'],['Original rewards','Donor-replaced rewards','Ablation effect']):
        for model,label in [('oracle','Oracle'),('rw','RW + softmax'),('gru','GRU'),('transformer','Transformer')]:
            line(ax,values(f'{model}_{metric}',True),label,COLORS[model])
        if metric!='delta_nll':ax.axhline(np.log(4),color='#999',ls=':',label='Uniform random')
        else:ax.axhline(0,color='#999',lw=.8)
        ax.set(title=title,ylabel='Delta NLL' if metric=='delta_nll' else 'Validation NLL')
    lo=min(ax.get_ylim()[0] for ax in axes[:2]);hi=max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:ax.set_ylim(lo,hi)
    finish(fig,axes,'B_donor')

if __name__=='__main__':plot()
