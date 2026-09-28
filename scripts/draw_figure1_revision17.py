"""Figure 1: the inferential question first, followed by a controlled test.

Reuses only original vector helpers and verified examples from the earlier
design. Bars in A are an input-operation example, not an oracle counterexample.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from draw_figure1_publication import (
    ROOT, REST, SPAT, text, box, line, arrow, token, tokens, lock,
    gru, attention, llama, INK, MUTED, EDGE, BLUE, GOLD,
    PALE_BLUE, PALE_GOLD, BG, PURPLE,
)
from paper_plot_style import COLORS

OUT = ROOT / 'outputs/figure1_revision17_20260924'
EXAMPLE = ROOT / 'outputs/figure1_publication_20260923/example_predictions.json'
SUMMARY = ROOT / 'outputs/task1_main_integrated_20260918/B_summary.json'


def heading(ax, letter, label, scale=1):
    text(ax,1,8*scale-1,letter,10,weight='bold')
    text(ax,12,8*scale-1,label,7.7,weight='bold')
    line(ax,1,18*scale-1,99,18*scale-1)


def section(ax, y, label):
    text(ax,2,y,label,6.2,color=MUTED,weight='bold')


def bars(ax, x, y, values, label):
    for j,p in enumerate(values):
        height=22*float(p)
        ax.add_patch(Rectangle((x+7*j,y-height),4.8,height,
            facecolor=BLUE if j==2 else '#BECBD5',edgecolor='none'))
        text(ax,x+7*j+2.4,y+6,'ABCD'[j],6.0,ha='center')
    text(ax,x+13,y+16,label,6.5,ha='center')


def panel_a(ax, example):
    heading(ax,'A','The inference we test')
    section(ax,28,'REPLACE REWARDS AT EVALUATION')
    text(ax,2,40,'Original rewards',6.6)
    tokens(ax,2,48,[('C','choice'),(66,'reward'),('C','choice'),(59,'reward')],w=11,h=10)
    text(ax,2,78,'Donor rewards',6.6)
    tokens(ax,2,86,[('C','choice'),(39,'reward'),('C','choice'),(44,'reward')],w=11,h=10)
    arrow(ax,54.5,53,69,70)
    arrow(ax,54.5,91,69,80)
    box(ax,70,61,28,30,BG,MUTED,r=2)
    lock(ax,90,57,.85)
    text(ax,84,76,'Model',7.2,ha='center',weight='bold')
    text(ax,50,110,'Same choices · fixed model parameters',6.3,ha='center')

    arrow(ax,50,121,50,135,MUTED)
    section(ax,148,'NEXT-CHOICE PROBABILITIES')
    bars(ax,9,181,example['intact'],'Original')
    bars(ax,64,181,example['donor'],'Donor')
    text(ax,50,217,'Choice C: 76% → 78%',7.1,ha='center')
    text(ax,50,232,'A small prediction change',6.7,ha='center')

    # The question mark marks the inference under scrutiny.
    ax.add_patch(FancyArrowPatch((50,241),(50,265),arrowstyle='-|>',
        mutation_scale=7,lw=.95,color=PURPLE))
    text(ax,62,253,'?',10,ha='center',color=PURPLE,weight='bold')
    box(ax,2,266,96,25,'#F4F0F7','#BCA7CA',r=2)
    text(ax,50,273,'Little reward dependence in',6.8,ha='center')
    text(ax,50,284,'the generating process?',7.0,ha='center',weight='bold')
    text(ax,50,299,'Does this inference follow?',7.1,ha='center',weight='bold')


def panel_b(ax, rest, spat):
    heading(ax,'B','Known ground truth')
    text(ax,50,28,'Synthetic agents, specified rules',6.8,ha='center',weight='bold')
    box(ax,1,39,98,92,'white',EDGE,r=2)
    box(ax,5,41,21,14,PALE_BLUE,BLUE,r=1,lw=.7)
    text(ax,15.5,48,'TASK A',6.0,ha='center',color=BLUE,weight='bold')
    text(ax,29,48,'Four-armed restless bandit',6.4,weight='bold')
    # Both mechanisms read the same past trials, but use different information.
    text(ax,39,57,'Past trials',6.2,ha='center',color=MUTED)
    text(ax,88,57,'Next',6.2,ha='center',color=MUTED)
    box(ax,9,62,61,20,'white',GOLD,r=2,lw=.95)
    tokens(ax,13,67,[('C','choice'),(66,'reward'),('C','choice'),(59,'reward')],w=11,h=10,gap=3)
    # Nested outlines select information in the same history, not duplicate inputs.
    box(ax,39,65,15,14,'none',BLUE,r=2,lw=1.1)
    arrow(ax,72,72,80,72)
    token(ax,83,67,'?','choice',11,10)
    line(ax,26,82,26,96,GOLD,.85)
    ax.plot([46.5,46.5,75,75],[79,88,88,96],color=BLUE,lw=.85,
            solid_joinstyle='round')
    text(ax,26,104,'Mechanism A',6.2,ha='center',color=GOLD,weight='bold')
    text(ax,75,104,'Mechanism B',6.2,ha='center',color=BLUE,weight='bold')
    text(ax,26,116,'Reward learning',6.4,ha='center',color=GOLD)
    text(ax,75,116,'Choice kernel',6.4,ha='center',color=BLUE)

    shift=0
    box(ax,1,141+shift,98,96,'white',EDGE,r=2)
    box(ax,5,143+shift,21,14,PALE_BLUE,BLUE,r=1,lw=.7)
    text(ax,15.5,150+shift,'TASK B',6.0,ha='center',color=BLUE,weight='bold')
    text(ax,29,150+shift,'Spatially correlated bandit',6.4,weight='bold')
    cue=int(spat['cue_arm'][0,0]); chosen=int(spat['actions'][0,0,0])
    values={cue:round(float(spat['cue_reward'][0,0])),chosen:round(float(spat['rewards'][0,0,0]))}
    for r in range(5):
        for c in range(5):
            k=r*5+c;x=7+c*7;y=162+r*7+shift
            box(ax,x,y,6.4,6.4,PALE_GOLD if k in values else BG,GOLD if k==chosen else EDGE,r=.25,lw=.5)
            if k in values: text(ax,x+3.2,y+3.2,str(values[k]),5.7,ha='center')
    ax.add_patch(Circle((7+(chosen%5)*7+3.2,162+(chosen//5)*7+3.2+shift),4.1,fill=False,ec=BLUE,lw=.75))
    text(ax,48,168+shift,'5 × 5 locations',6.5)
    text(ax,48,181+shift,'Cue + selected',6.3)
    text(ax,48,191+shift,'feedback',6.3)
    x=np.linspace(10,40,50)
    ax.plot(x,215+shift-9*np.exp(-((x-25)/9)**2),color=GOLD,lw=1)
    ax.add_patch(Circle((25,207+shift),1.5,fc=GOLD,ec='none'))
    arrow(ax,76,210+shift,66,210+shift,BLUE);arrow(ax,80,210+shift,91,210+shift,BLUE)
    arrow(ax,78,208+shift,78,202+shift,BLUE)
    ax.add_patch(Circle((78,210+shift),2.1,fc=BLUE,ec='white',lw=.5))
    text(ax,25,221+shift,'Mechanism A',6.0,ha='center',color=GOLD,weight='bold')
    text(ax,77,221+shift,'Mechanism B',6.0,ha='center',color=BLUE,weight='bold')
    text(ax,25,230+shift,'Generalize rewards',6.2,ha='center',color=GOLD)
    text(ax,77,230+shift,'Stay nearby',6.4,ha='center',color=BLUE)

    text(ax,50,245+shift,'Reward weight shifts the balance',6.4,ha='center',weight='bold')
    text(ax,24,256+shift,'Choice-history-led',6.0,ha='center',color=BLUE)
    text(ax,77,256+shift,'Reward-learning-led',6.0,ha='center',color=GOLD)
    # The colour key ties the weighting manipulation to the two mechanism types.
    for k in range(88):
        blend=k/87
        colour=np.array(to_rgb(BLUE))*(1-blend)+np.array(to_rgb(GOLD))*blend
        line(ax,6+k,264+shift,7+k,264+shift,colour,1.6)
    for w in (0,.1,.3,.5,.7,.9,1):
        xx=6+88*w
        ax.add_patch(Circle((xx,264+shift),1.7,fc=INK,ec='white',lw=.5))
        text(ax,xx,274+shift,'0' if w==0 else ('1' if w==1 else str(w)[1:]),6.0,ha='center')
    arrow(ax,50,279+shift,50,283+shift)
    box(ax,2,285+shift,96,15,BG,EDGE,r=2)
    text(ax,50,292.5+shift,'Generate synthetic trajectories',6.8,ha='center',weight='bold')
    ax.set_ylim(304,-1)


def symbolic(ax,x,y):
    """A rule card, distinct from the architecture-specific neural icons."""
    c=COLORS['rw']
    box(ax,x-8,y-9,16,18,'white',c,r=1,lw=.8)
    text(ax,x,y-3,'f(x)',7.6,ha='center',color=c)
    line(ax,x-5,y+3,x+1,y+3,c,.75)
    arrow(ax,x+1,y+3,x+5,y+3,c,.75)
    line(ax,x-5,y+6,x+5,y+6,c,.6)


def panel_c(ax, summary):
    heading(ax,'C','Validate the response')
    box(ax,1,23,98,96,'white',EDGE,r=2)
    text(ax,50,32,'Fit models to synthetic trajectories',6.7,ha='center',weight='bold')
    gru(ax,25,48,.8);attention(ax,75,48,.7)
    text(ax,25,61,'GRU',6.5,ha='center',weight='bold')
    text(ax,75,61,'Transformer',6.3,ha='center',weight='bold')
    llama(ax,25,80,.65);symbolic(ax,75,80)
    text(ax,25,95,'LLaMA SFT',6.3,ha='center',weight='bold')
    text(ax,75,95,'Symbolic models',6.3,ha='center',weight='bold')
    text(ax,75,104,'RW / fitted generator',5.6,ha='center',color=MUTED)
    text(ax,50,113,'Neural: full input / choice-only fits',6.0,ha='center')

    text(ax,50,130,'Apply the same replacement',6.8,ha='center',weight='bold')
    text(ax,2,145,'Original',6.3)
    tokens(ax,34,140,[('C','choice'),(66,'reward'),('C','choice'),(59,'reward')],w=11,h=9,gap=3)
    text(ax,2,160,'Donor',6.3)
    tokens(ax,34,155,[('C','choice'),(39,'reward'),('C','choice'),(44,'reward')],w=11,h=9,gap=3)
    text(ax,50,175,'Choices held fixed · fitted parameters frozen',5.9,ha='center')
    arrow(ax,50,180,50,186,MUTED)

    # Real group means from the same restless-task w=1 validation data and
    # the same 20 donor-reward assignments. The two token rows above only
    # illustrate the edit; the bars are not single-trial probabilities.
    box(ax,2,190,96,99,'white',EDGE,r=2)
    text(ax,50,198,'ONE GENERATOR · w = 1',6.2,
         ha='center',weight='bold')
    text(ax,50,208,'Restless donor effect (nats / choice)',6.0,
         ha='center',color=MUTED)
    values=[('oracle','Oracle'),('fit','Fitted gen.'),('rw','RW + softmax'),
            ('gru','GRU'),('transformer','Transformer'),('llama','LLaMA SFT')]
    x0,x1=40,82
    maximum=summary['oracle_delta_nll']['mean']
    for j,(key,name) in enumerate(values):
        val=summary[key+'_delta_nll']['mean']
        y=219+12*j
        text(ax,4,y,name,6.0,weight='bold' if j==0 else 'normal')
        box(ax,x0,y-3.5,x1-x0,6.5,'#F0F2F3','none',r=.7,lw=0)
        box(ax,x0,y-3.5,(x1-x0)*val/maximum,6.5,COLORS[key],'none',r=.7,lw=0)
        text(ax,86,y,f'{val:.2f}',6.2)
    line(ax,4,225,94,225,EDGE,.45)
    text(ax,50,297,'Same data, different ablation responses',6.4,
         ha='center',weight='bold',color=PURPLE)


def setup(ax):
    ax.set(xlim=(-1,101),ylim=(304,-1));ax.axis('off')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    with np.load(REST) as z:rest=dict(z)
    with np.load(SPAT) as z:spat=dict(z)
    example=json.loads(EXAMPLE.read_text())
    summary=json.loads(SUMMARY.read_text())['1.0']
    draws=[lambda ax:panel_a(ax,example),lambda ax:panel_b(ax,rest,spat),
           lambda ax:panel_c(ax,summary)]
    fig=plt.figure(figsize=(5.5,4.95))
    fig.text(.5,.975,'Does model ablation reveal how choices were generated?',
        ha='center',va='center',fontsize=9,fontweight='bold',color=INK)
    for j,draw in enumerate(draws):
        ax=fig.add_axes([.01+j*.335,.015,.318,.925]);setup(ax);draw(ax)
    for xx in (.332,.667):
        fig.add_artist(plt.Line2D([xx,xx],[.035,.935],transform=fig.transFigure,color=EDGE,lw=.7))
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'Figure1_design.{ext}',dpi=350)
    plt.close(fig)
    for letter,draw in zip('ABC',draws):
        fig=plt.figure(figsize=(5.5*.318,4.95*.925))
        ax=fig.add_axes([0,0,1,1]);setup(ax);draw(ax)
        for ext in ('pdf','svg','png'):fig.savefig(OUT/f'Figure1_panel_{letter}.{ext}',dpi=350)
        plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps({
        'design':'Problem -> known synthetic generators -> matched response validation',
        'data_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REST,SPAT,EXAMPLE,SUMMARY)},
        'example_scope':'A uses a generic fitted-model diagram for evaluation-time replacement. Its probability example comes from one fitted Transformer trial, not a model-oracle disagreement. C token rows illustrate the donor operation; its six bars are empirical mean effects read directly from the audited restless-task w=1 summary. They are not single-trial probabilities.',
        'fit_regimes':'Neural families and RW are fitted to pooled training histories; fitted-generator parameters are calibrated per validation participant on an earlier prefix. Only neural families have separate full-input and choice-only training. The oracle is the true-parameter generator, not a fitted predictor.',
        'science_boundary':'Conditional reward-history replacement with actual choices fixed, not new behavior after changing the environment.',
        'generator_formulas':'Kept in manuscript Methods; removed from the overview to reduce unexplained notation.',
        'history_mechanisms':'Restless reward learning uses paired past choices and rewards; its signed one-step kernel uses only the last choice. Weight changes the generating rule, not a directly prescribed ablation magnitude.',
    },indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':main()
