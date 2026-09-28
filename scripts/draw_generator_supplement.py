"""Existing trained-torso GRU teacher sweep, Figure 3 companion."""
import json
import numpy as np
import matplotlib.pyplot as plt
import draw_iclr_main_drafts as d
ROOT=d.Path('outputs/neural_generator_sweep_v1')

def main():
    d.style();fig,axs=plt.subplots(2,2,figsize=(7.1,4.8))
    for ax,k,title,letter in zip(axs.flat,('intact_nll','donor_nll','delta_nll','response_error'),
        ('Original rewards','Donor-replaced rewards','Ablation effect','Response deviation from oracle'),'abcd'):
        for m in ('oracle','rw','gru','transformer'):
            rows=np.array([[np.load(ROOT/f'head{h}_w{round(w*100):03d}'/'B_vectors.npz')[m+'_'+k].mean() for w in d.W] for h in (101,102,103)])
            assert np.isfinite(rows).all()
            ax.plot(d.W,rows.mean(0),color=d.COLORS[m],marker=d.MARK[m],ms=3.2,label=d.LABEL[m])
        d.panel(ax,letter,title);ax.set_xlabel('Reward-visible policy mixture weight')
        ax.set_ylabel('Validation choice NLL (nats)' if k.endswith('nll') and k!='delta_nll' else ('$\\Delta$NLL (donor $-$ original)' if k=='delta_nll' else 'Probability-response error'))
        if k in ('intact_nll','donor_nll'):ax.axhline(np.log(4),ls=':',lw=.8,color='#999')
        else:ax.axhline(0,color='#aaa',lw=.65)
    lo=min(ax.get_ylim()[0] for ax in axs[0]);hi=max(ax.get_ylim()[1] for ax in axs[0])
    for ax in axs[0]:ax.set_ylim(lo,hi)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
    fig.text(.5,.91,'Frozen trained GRU torso + reset output head | mean across 3 generator seeds',ha='center',fontsize=7)
    fig.subplots_adjust(left=.1,right=.985,bottom=.105,top=.79,hspace=.65,wspace=.32)
    d.save(fig,'FigureS2_GRU_generator',list(axs.flat))
    (d.OUT/'FigureS2_caption.md').write_text('''# Figure S2: sensitivity to generator construction

Frozen previously trained GRU torso with independently reset output heads (seeds 101, 102, 103). Seven mixtures of reward-visible and constant-reward policies. Curves average the three generator seeds; individual-seed curves and uncertainty bands are not shown. Each cell uses 500 training and 100 validation sessions, 200 trials; B scores trials 151-200 and 20 matched donor permutations. Real choices remain fixed. Fresh pooled RW, GRU and Transformer students; one neural training seed per cell. No LLaMA in this experiment.

This teacher inherits its torso from symbolic-data training. The experiment is a generator-construction sensitivity control, not a fully random teacher or a causal isolation of architecture matching. The mixture weight differs conceptually from the psychological reward policy weight in the original task. Student parameters are not shared with the teacher. Validation, not sealed-test confirmation. All 21 cells retained. Fully random GRU and Transformer controls are being run separately; no pending results included here.
''',encoding='utf-8')

if __name__=='__main__':main()
