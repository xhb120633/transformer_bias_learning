"""Publication-sized vector drafts from audited summaries. No fitting."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from paper_plot_style import COLORS, weight_axis, WEIGHT_TICKS

OUT=Path('outputs/iclr_main_figures_draft_20260921')
T1=Path('outputs/task1_main_integrated_20260918')
T2=Path('outputs/spatial_main_fixedshape_integrated_20260921')
SPLIT_LABEL='Validation'
W=np.array([0,.1,.3,.5,.7,.9,1.])
LABEL={'oracle':'Oracle','fit':'Fitted generator','rw':'RW + softmax','gru':'GRU','transformer':'Transformer','llama':'LLaMA SFT'}
MARK={'oracle':'o','fit':'s','rw':'D','gru':'^','transformer':'v','llama':'o'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
      'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
      'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
      'lines.linewidth':1.45,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
      'xtick.major.width':.6,'ytick.major.width':.6,'savefig.facecolor':'white'})

def curve(ax,rows,m):
    ci=np.array([r['ci95'] for r in rows]);y=[r['mean'] for r in rows]
    ax.fill_between(W,ci[:,0],ci[:,1],color=COLORS[m],alpha=.10,lw=0)
    ax.plot(W,y,color=COLORS[m],marker=MARK[m],ms=3.2,mew=.5,
      ls='--' if m=='fit' else '-',label=LABEL[m])

def panel(ax,letter,title):
    ax.set_title(r'$\bf{'+letter+r'}$  '+title,loc='left',pad=10)
    weight_axis(ax)
    ax.tick_params(length=3)

def save(fig,name,axes=()):
    fig.canvas.draw()
    for ax in axes: np.testing.assert_allclose(ax.get_xticks(),WEIGHT_TICKS)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=260)
    plt.close(fig)

def figure2():
    a=json.loads((T1/'A_summary.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.85),sharey=True)
    for ax,mode,letter,title in zip(axs,('full','choice'),('a','b'),('Full-input training','Choice-only training')):
        for m in ('gru','transformer','llama'):curve(ax,[a[str(float(w))]['all'][m+'_'+mode+'_nll'] for w in W],m)
        curve(ax,[a[str(float(w))]['all']['oracle_nll'] for w in W],'oracle')
        ax.axhline(np.log(4),color='#999999',lw=1,ls=':',label='Uniform random')
        panel(ax,letter,title);ax.set_ylim(.27,1.47)
    axs[0].set_ylabel(f'{SPLIT_LABEL} choice NLL (nats)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.53,1.0),handlelength=2)
    fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.76,wspace=.18)
    save(fig,'Figure2_prediction',axs)

def figure_b(root,num):
    b=json.loads((root/'B_summary.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.7))
    metrics=('intact_nll','donor_nll','delta_nll','response_error')
    titles=('Original rewards','Donor-replaced rewards','Ablation effect','Response deviation from oracle')
    for ax,k,letter,title in zip(axs.flat,metrics,'abcd',titles):
        for m in LABEL:curve(ax,[b[str(float(w))][m+'_'+k] for w in W],m)
        panel(ax,letter,title)
        if k in ('intact_nll','donor_nll'):
            ax.axhline(np.log(4 if num==3 else 25),color='#999',lw=.85,ls=':')
            ax.set_ylabel(f'{SPLIT_LABEL} choice NLL (nats)')
        elif k=='delta_nll':
            ax.set_ylabel('$\\Delta$NLL (donor $-$ original)');ax.axhline(0,color='#aaa',lw=.65)
        else:
            ax.set_ylabel('Probability-response error');ax.axhline(0,color='#aaa',lw=.65)
    lo=min(ax.get_ylim()[0] for ax in axs[0]);hi=max(ax.get_ylim()[1] for ax in axs[0])
    for ax in axs[0]:ax.set_ylim(lo,hi)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,1),handlelength=2.1,columnspacing=2)
    fig.subplots_adjust(left=.09,right=.985,bottom=.105,top=.80,hspace=.64,wspace=.32)
    save(fig,f'Figure{num}_ablation',list(axs.flat))

def figure1():
    fig=plt.figure(figsize=(7.1,5.0));ax=fig.add_axes([.025,.035,.95,.93]);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    ink='#303640';muted='#657180';edge='#cbd2da';bg='#f5f7f9'
    def text(x,y,s,size=8,**kw):ax.text(x,y,s,fontsize=size,color=ink,va='center',**kw)
    def box(x,y,w,h,s='',color=bg,size=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=1.2',fc=color,ec=edge,lw=.7))
        if s:text(x+w/2,y+h/2,s,size,ha='center',linespacing=1.45)
    def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',lw=.9,color=muted,mutation_scale=9))
    text(0,98,'a',11,fontweight='bold');text(4,98,'Known generators, controlled reward contributions',9,fontweight='bold')
    box(1,72,46,21);box(53,72,46,21)
    text(4,89,'Restless bandit',8.5,fontweight='bold')
    for i,c in enumerate('ABCD'):
        box(4+i*6,78,4.7,6,c,color='white',size=7)
    text(4,75,'Reward learning + choice kernel',7)
    text(56,89,'Spatial bandit',8.5,fontweight='bold')
    for r in range(5):
        for c in range(5):
            ax.add_patch(Rectangle((56+c*1.8,76+r*1.8),1.55,1.55,fc=('#228b78' if (r,c)==(2,2) else '#e2e8ec'),ec='white',lw=.3))
    text(68,81,'Reward generalization\n+ local search',7.5,linespacing=1.4)
    text(50,66,'Generating reward weight  w = 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1',8,ha='center')
    arrow(50,62,50,58)
    text(0,55,'b',11,fontweight='bold');text(4,55,'Fit behavior, then evaluate held-out participants',9,fontweight='bold')
    box(1,37,27,13,'Synthetic histories\nchoice, reward, choice, reward, ...',size=7.3)
    arrow(29,43.5,34,43.5)
    box(35,35,38,17,'GRU / Transformer / LLaMA SFT\nNeural nets: full input or choice-only\nRW + generator fits: full input',size=7.5)
    arrow(74,43.5,79,43.5)
    box(80,37,19,13,'One-step-ahead\nchoice probabilities',size=7.3)
    text(50,30,'Causal histories; real past choices retained; no subject embeddings',7,ha='center')
    text(0,24,'c',11,fontweight='bold');text(4,24,'Separate predictive performance from ablation-response fidelity',9,fontweight='bold')
    box(1,1,29,18,'Training-input comparison\nFull input vs. choice-only\nPredictive NLL',color='#eef4f3',size=7.5)
    box(35,1,34,18,'Matched evaluation intervention\nReplace past rewards with donor rewards\nKeep real choices fixed',size=7)
    arrow(70,10,74,10)
    box(75,1,24,18,'Model vs. oracle response\nCompare changes in NLL\nand choice probabilities',color='#f1eef6',size=7)
    save(fig,'Figure1_design')

def main():
    assert json.loads((T2/'COMPLETE.json').read_text())['scientific_audit_passed']
    OUT.mkdir(exist_ok=True);style();figure1();figure2();figure_b(T1,3);figure_b(T2,4)
    sources=[T1/'A_summary.json',T1/'B_summary.json',T2/'B_summary.json',T2/'COMPLETE.json']
    (OUT/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
    print(OUT.resolve())

if __name__=='__main__':main()
