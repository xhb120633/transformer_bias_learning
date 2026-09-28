"""Editable, evidence-grounded Figure 1; each panel can also be exported alone."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc, PathPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/figure1_publication_20260923'
REST = ROOT / 'outputs/weight_curve_a_20260915/reward_w050/val_000.npz'
TRAIN = ROOT / 'outputs/weight_curve_a_20260915/reward_w050/train_000.npz'
MAPS = ROOT / 'outputs/weight_curve_a_20260915/reward_w050/validation_assay.npz'
SPAT = ROOT / 'outputs/spatial_pilot_v1/reward_w050/val_audit.npz'
PRED = ROOT / 'outputs/weight_neural_full_20260916/transformer_reward_w050_full_seed11/assay_full.npz'
INK='#26313D'; MUTED='#637383'; EDGE='#D6DEE5'; BLUE='#447CA3'; GOLD='#BD7927'
PALE_BLUE='#EAF2F7'; PALE_GOLD='#FBF0DE'; BG='#F5F7F9'; PURPLE='#8054A1'
MODEL={'gru':'#228B78','transformer':'#536EBA','llama':PURPLE}
TEXT=[]


def text(ax,x,y,s,size=7,ha='left',color=INK,weight='normal',**kw):
    obj=ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,
                fontweight=weight,linespacing=1.15,**kw)
    TEXT.append(obj)
    return obj


def box(ax,x,y,w,h,fill='white',edge=EDGE,r=1.5,lw=.65):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=fill,edgecolor=edge,linewidth=lw)
    ax.add_patch(p);return p


def line(ax,x1,y1,x2,y2,c=EDGE,lw=.7,**kw):
    ax.plot([x1,x2],[y1,y2],color=c,lw=lw,solid_capstyle='round',**kw)


def arrow(ax,x1,y1,x2,y2,c=MUTED,lw=.75,rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
        mutation_scale=5.8,lw=lw,color=c,connectionstyle=f'arc3,rad={rad}',
        shrinkA=0,shrinkB=0))


def title(ax,letter,name):
    text(ax,1,6,letter,10,weight='bold')
    text(ax,12,6,name,8.1,weight='bold')
    line(ax,1,15,99,15)


def token(ax,x,y,label,kind='choice',width=12,height=10):
    c=BLUE if kind=='choice' else GOLD
    fill=PALE_BLUE if kind=='choice' else PALE_GOLD
    if kind=='mask':c=MUTED;fill=BG
    box(ax,x,y,width,height,fill,c,r=2 if kind=='choice' else .5,lw=.6)
    text(ax,x+width/2,y+height/2,str(label),6.9,ha='center')


def tokens(ax,x,y,seq,gap=2.5,w=12,h=10):
    for value,kind in seq:
        token(ax,x,y,value,kind,w,h);x+=w+gap
    return x-gap


def person(ax,x,y,scale=1,c=MUTED):
    ax.add_patch(Circle((x,y),2.4*scale,fc='white',ec=c,lw=.8))
    ax.add_patch(Arc((x,y+7*scale),10*scale,9*scale,theta1=180,theta2=360,ec=c,lw=.8))
    line(ax,x-5*scale,y+7*scale,x+5*scale,y+7*scale,c)


def lock(ax,x,y,scale=1,c=MUTED):
    ax.add_patch(Arc((x,y-1*scale),5*scale,7*scale,theta1=180,theta2=360,ec=c,lw=.7))
    box(ax,x-4*scale,y,8*scale,6*scale,'white',c,r=.8)
    ax.add_patch(Circle((x,y+2.6*scale),.65*scale,fc=c,ec='none'))


def gru(ax,x,y,scale=1,c=MODEL['gru']):
    for dx in (-7,0,7):
        ax.add_patch(Circle((x+dx*scale,y),2.6*scale,fc='white',ec=c,lw=.9))
    arrow(ax,x-4*scale,y,x-2.5*scale,y,c)
    arrow(ax,x+2.6*scale,y,x+4.5*scale,y,c)
    ax.add_patch(FancyArrowPatch((x+6*scale,y-3*scale),(x-6*scale,y-3*scale),
        connectionstyle='arc3,rad=.8',arrowstyle='-|>',mutation_scale=5,lw=.8,color=c))


def attention(ax,x,y,scale=1,c=MODEL['transformer']):
    for k in (2,1,0):
        box(ax,x-8*scale+k*1.5*scale,y-7*scale-k*1.5*scale,16*scale,14*scale,
            'white',c,r=.7)
    for r in range(3):
        for col in range(3):
            if col<=r:
                ax.add_patch(Rectangle((x-5.5*scale+col*4*scale,y-4.5*scale+r*4*scale),
                    2.9*scale,2.9*scale,fc=c,ec='none',alpha=.3+.2*(r==col)))


def llama(ax,x,y,scale=1):
    c=MODEL['llama']
    for k in (2,1,0):
        box(ax,x-10*scale+k*1.8*scale,y-8*scale-k*1.7*scale,20*scale,15*scale,
            'white',c,r=1.1)
    for yy in (-3,1,5):line(ax,x-6*scale,y+yy*scale,x+3*scale,y+yy*scale,c,.65)
    box(ax,x+4*scale,y-2*scale,11*scale,8*scale,PALE_GOLD,GOLD,r=1)
    text(ax,x+9.5*scale,y+2*scale,'+',7,ha='center',color=GOLD,weight='bold')


def modelbox(ax,x,y,label,lockit=False):
    box(ax,x,y,20,15,BG,MUTED,r=2)
    text(ax,x+10,y+8,label,7,ha='center')
    if lockit:lock(ax,x+15,y-2,.65)


def rule_label(ax,x,y,s,c):
    line(ax,x,y-3,x+3,y-3,c,1.5)
    text(ax,x+5,y-3,s,6.6)


def panel_a(ax,rest,spat):
    title(ax,'A','Tasks & generators')
    text(ax,1,24,'Four-armed restless bandit',7.4,weight='bold')
    for j,a in enumerate('ABCD'):
        x=7+23*j
        box(ax,x,32,16,15,PALE_BLUE if a=='C' else 'white',BLUE if a=='C' else EDGE,r=2)
        text(ax,x+8,39.5,a,7.7,ha='center',weight='bold' if a=='C' else 'normal')
    arrow(ax,61,48,61,53,GOLD)
    token(ax,52,54,66,'reward',18,10)
    text(ax,3,54,'Chosen reward',6.4)
    # A real example of the latent environment, displayed only as an environment schematic.
    xx=np.linspace(10,89,200)
    for j,c in enumerate(('#8796A5','#7798B1','#BD7927','#A3AFB9')):
        yy=77-(rest['latent_reward_mean'][0,:,j]/100)*10
        ax.plot(xx,yy,lw=.6,color=c,alpha=.9)
    text(ax,51,82,'Hidden reward means drift',6.2,ha='center',color=MUTED)
    # Keep the observed-feedback and choice paths outside the latent schedule.
    # The hidden reward means are not an input to the generating learner.
    ax.plot([52,3,3,23],[60,60,88,88],color=GOLD,lw=.75,solid_joinstyle='round')
    arrow(ax,23,88,23,91,GOLD)
    ax.plot([69,72,98,98,79],[47,50,50,88,88],color=BLUE,lw=.75,solid_joinstyle='round')
    arrow(ax,79,88,79,91,BLUE)
    text(ax,23,96,r'$\Delta Q(a_t)=\alpha\delta_t$',7.0,ha='center')
    text(ax,79,96,r'$\pm\,K_{\rm last}$',7.3,ha='center')
    text(ax,23,106,'Reward learning',6.5,ha='center')
    text(ax,77,106,'Repeat / switch',6.5,ha='center')
    line(ax,7,111,42,111,GOLD,1.5);line(ax,57,111,94,111,BLUE,1.5)
    text(ax,50,120,r'$u_t=w\,\widetilde{Q}_t+(1-w)K_t$',8,ha='center')
    text(ax,50,129,'Greedy choice + 10% choice noise',6.1,ha='center',color=MUTED)
    line(ax,1,136,99,136)
    text(ax,1,145,'Spatially correlated bandit',7.4,weight='bold')
    cue=int(spat['cue_arm'][0,0]);chosen=int(spat['actions'][0,0,0])
    values={cue:round(float(spat['cue_reward'][0,0])),chosen:round(float(spat['rewards'][0,0,0]))}
    for r in range(5):
        for c in range(5):
            k=r*5+c;x=4+c*7;y=155+r*7
            fill=PALE_GOLD if k in values else BG
            box(ax,x,y,6.5,6.5,fill,GOLD if k==chosen else EDGE,r=.25,lw=.5)
            if k in values:text(ax,x+3.25,y+3.25,str(values[k]),5.3,ha='center')
    ax.add_patch(Circle((4+(chosen%5)*7+3.25,155+(chosen//5)*7+3.25),4.3,
                        fill=False,ec=BLUE,lw=.8))
    text(ax,48,158,'5 × 5 locations',6.7)
    text(ax,48,168,'Cue + chosen reward',6.3)
    text(ax,48,180,'8 maps × 20 choices',6.3,color=MUTED)
    # Mechanism icons: a smooth generalization kernel and local directional steps.
    xx=np.linspace(8,40,50);yy=203-9*np.exp(-((xx-24)/9)**2)
    ax.plot(xx,yy,color=GOLD,lw=1);ax.add_patch(Circle((24,195),1.4,fc=GOLD,ec='none'))
    arrow(ax,77,201,65,201,BLUE);arrow(ax,79,199,79,190,BLUE)
    arrow(ax,81,201,93,201,BLUE);ax.add_patch(Circle((79,201),2.1,fc=BLUE,ec='white',lw=.5))
    text(ax,24,211,'Generalize rewards',6.3,ha='center')
    text(ax,78,211,'Stay nearby',6.4,ha='center')
    text(ax,50,224,r'$\pi_t=w\,\pi_R+(1-w)\pi_L$',8,ha='center')
    text(ax,50,233,'Policy mixture + 10% choice noise',6.1,ha='center',color=MUTED)
    # The same seven manipulation values, with separate equations above.
    line(ax,5,246,95,246,EDGE,1.2)
    for w in (0,.1,.3,.5,.7,.9,1):
        x=5+90*w;ax.add_patch(Circle((x,246),1.65,fc=INK,ec='white',lw=.5))
        text(ax,x,255,'0' if w==0 else ('1' if w==1 else str(w)[1:]),5.9,ha='center')
    text(ax,50,264,'Generating reward weight  w',6.5,ha='center')


def panel_b(ax,train,pred):
    title(ax,'B','Fit & evaluate')
    text(ax,1,24,'Pooled participant histories',7.2,weight='bold')
    for j in range(3):
        y=37+13*j;person(ax,7,y+1,.85)
        seq=[(str('ABCD'[int(train['action'][j,0])]),'choice'),(int(train['reward'][j,0]),'reward'),
             (str('ABCD'[int(train['action'][j,1])]),'choice'),(int(train['reward'][j,1]),'reward')]
        tokens(ax,19,y-3,seq,w=11,h=9)
        text(ax,80,y+1,'…',8)
    arrow(ax,49,76,49,86)
    text(ax,49,94,r'$\mathcal{L}=-\sum_t\log p_\theta(a_t\mid h_t)$',8,ha='center')
    text(ax,49,104,'Loss on choice targets only',6.6,ha='center')
    arrow(ax,49,110,49,116)
    # Parallel model icons; the small networks start from scratch.
    gru(ax,23,124,1.08);attention(ax,74,124,1.05)
    text(ax,23,139,'GRU',7.2,ha='center',weight='bold')
    text(ax,74,139,'Transformer',7.2,ha='center',weight='bold')
    text(ax,49,148,'Trained from scratch',6.4,ha='center',color=MUTED)
    llama(ax,22,168,1.05)
    text(ax,47,162,'LLaMA SFT',7.2,weight='bold')
    text(ax,47,173,'Base + LoRA',7.0)
    line(ax,1,191,99,191)
    text(ax,1,201,'Validation choice prediction',7.0,weight='bold')
    person(ax,8,215,.9)
    tokens(ax,18,210,[('C','choice'),('66','reward'),('C','choice'),('59','reward')],w=9,h=9)
    arrow(ax,65,215,74,215)
    modelbox(ax,77,207,r'$f_\theta$',True)
    ax.plot([87,87,31],[225,229,229],color=MUTED,lw=.75,solid_joinstyle='round')
    arrow(ax,31,229,31,232)
    probs=pred['intact_probabilities'][0,2]
    # An actual fitted Transformer prediction for the same example third trial.
    for j,p in enumerate(probs):
        x=8+10*j;hh=16*float(p)
        ax.add_patch(Rectangle((x,247-hh),6,hh,fc=BLUE if j==2 else '#BBC9D4',ec='none'))
        text(ax,x+3,253,'ABCD'[j],5.8,ha='center')
    text(ax,62,239,r'$p(a_t\mid h_t)$',7,ha='center')
    text(ax,62,250,'Score: NLL',6.5,ha='center')
    text(ax,49,264,'Oracle · fitted generator · RW',6.2,ha='center',color=MUTED)


def panel_c(ax,donor_rewards,example):
    title(ax,'C','Manipulate information')
    text(ax,1,24,'Training: separate fits',7.2,weight='bold')
    text(ax,1,36,'Full input',6.8)
    tokens(ax,1,44,[('C','choice'),(66,'reward'),('C','choice'),(59,'reward')],w=12,h=10)
    arrow(ax,62,49,73,49)
    modelbox(ax,77,41,r'$\theta_{\rm full}$')
    text(ax,1,66,'Choice only',6.8)
    tokens(ax,1,74,[('C','choice'),('C','choice')],w=12,h=10)
    arrow(ax,31,79,73,79)
    modelbox(ax,77,71,r'$\theta_{\rm choice}$')
    # A one-trial crop of the real natural-language serialization.
    box(ax,1,92,97,49,'white',EDGE,r=2)
    text(ax,6,99,'LLaMA SFT prompt excerpt',6.3,weight='bold')
    text(ax,6,108,'Full',5.8,color=MUTED)
    text(ax,24,108,'You press <<C>>',6.0,family='DejaVu Sans Mono')
    box(ax,22,112,71,8,PALE_GOLD,'none',r=.3,lw=0)
    text(ax,24,116,'and receive 66 points.',5.7,family='DejaVu Sans Mono')
    text(ax,6,127,'Choice only',5.8,color=MUTED)
    text(ax,6,136,'You press <<C>>.',6.2,family='DejaVu Sans Mono')
    text(ax,1,147,'Evaluation: fixed full-input model',6.8,weight='bold')
    text(ax,1,160,'Original rewards',6.5)
    tokens(ax,1,168,[('C','choice'),(66,'reward'),('C','choice'),(59,'reward')],w=12,h=10)
    text(ax,1,187,'Donor rewards',6.5)
    tokens(ax,1,195,[('C','choice'),(int(donor_rewards[0]),'reward'),('C','choice'),(int(donor_rewards[1]),'reward')],w=12,h=10)
    arrow(ax,62,173,78,183)
    arrow(ax,62,200,78,190)
    modelbox(ax,78,178,r'$f_\theta$',True)
    text(ax,2,215,'Prompt: receive',6.3)
    token(ax,49,211,'66','reward',11,8);arrow(ax,63,215,70,215,GOLD)
    token(ax,74,211,'39','reward',11,8)
    text(ax,91,215,'pts',5.8,ha='center',color=MUTED)
    for x0,key,label in [(7,'intact','Original'),(60,'donor','Donor')]:
        for j,p in enumerate(example[key]):
            ax.add_patch(Rectangle((x0+j*7,246-18*p),4.8,18*p,
                fc=BLUE if j==2 else '#BBC9D4',ec='none'))
            text(ax,x0+j*7+2.4,251,'ABCD'[j],5.3,ha='center')
        text(ax,x0+13,259,label,6.2,ha='center')
    arrow(ax,36,238,53,238,GOLD)
    text(ax,45,229,r'$\Delta p$',7,ha='center')
    text(ax,50,270,r'$\Delta\mathrm{NLL}\ ;\quad\Delta p_{\rm model}\leftrightarrow\Delta p_{\rm oracle}$',7.0,ha='center')


def setup(ax):
    ax.set(xlim=(-1,101),ylim=(276,-1));ax.axis('off')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans',
        'savefig.facecolor':'white'})
    with np.load(REST) as z:rest=dict(z)
    with np.load(TRAIN) as z:train=dict(z)
    with np.load(SPAT) as z:spat=dict(z)
    with np.load(PRED) as z:pred=dict(z)
    np.testing.assert_array_equal(pred['participants'],rest['base_participant_id'])
    with np.load(MAPS) as z:donor=int(z['donor_mappings'][0,0])
    dr=rest['reward'][donor,:2]
    example=json.loads((OUT/'example_predictions.json').read_text())
    assert rest['action'][0,:3].tolist()==[2,2,2]
    assert rest['reward'][0,:2].tolist()==[66,59] and dr.tolist()==[39,44]
    draws=[lambda ax:panel_a(ax,rest,spat),lambda ax:panel_b(ax,train,pred),lambda ax:panel_c(ax,dr,example)]
    fig=plt.figure(figsize=(5.5,4.7))
    for j,draw in enumerate(draws):
        ax=fig.add_axes([.01+j*.335,.015,.318,.97]);setup(ax);draw(ax)
    for xx in (.332,.667):
        fig.add_artist(plt.Line2D([xx,xx],[.04,.94],transform=fig.transFigure,color=EDGE,lw=.65))
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'Figure1_design.{ext}',dpi=350)
    plt.close(fig)
    for letter,draw in zip('ABC',draws):
        fig=plt.figure(figsize=(5.5*.318,4.7*.97));ax=fig.add_axes([0,0,1,1]);setup(ax);draw(ax)
        for ext in ('pdf','svg','png'):fig.savefig(OUT/f'Figure1_panel_{letter}.{ext}',dpi=350)
        plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps({
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TRAIN,REST,MAPS,SPAT,PRED]},
        'example':{'recipient_index':0,'donor_index':donor,'choices':[2,2,2],
                   'rewards':[66,59],'donor_rewards':dr.tolist()},
        'visual_scope':'Original vector task/method illustrations; probability bars show the saved fitted Transformer prediction for the illustrated history.',
        'design_references':['https://www.nature.com/documents/Nature_scientific_illustration_author_guide.pdf',
                             'https://www.nature.com/articles/s41562-025-02324-0',
                             'https://charleywu.github.io/downloads/wu2018exploration.pdf'],
        'external_skill_reviewed':'https://github.com/shreyas-makes/science-illustrations-skill (reviewed only; no raster workflow installed)',
    },indent=2))
    print(OUT)


if __name__=='__main__':main()
