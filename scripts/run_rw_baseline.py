"""Train-only pooled RW-softmax baseline, matched validation donor evaluation."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from ablate_spatial_references import summarize, nll
from run_weight_curve_a import replay as oracle_replay

OUT=Path('outputs/rw_pooled_baseline_v1')
WEIGHTS=[0.,.1,.3,.5,.7,.9,1.]


def rw(theta,actions,rewards,arms,cue=None,gradient=False):
    """Q0=50 raw points, fixed affine reward units; analytical likelihood gradient."""
    alpha,logbeta=theta;beta=np.exp(logbeta)
    rewards=(rewards-50.)/100.
    n,tmax=actions.shape;q=np.zeros((n,arms));dq=np.zeros_like(q);idx=np.arange(n)
    total=0.;grad=np.zeros(2);probs=[]
    def update(a,r):
        error=r-q[idx,a]
        dq[idx,a]=(1-alpha)*dq[idx,a]+error
        q[idx,a]+=alpha*error
    if cue is not None:update(cue[0],(cue[1]-50.)/100.)
    for t in range(tmax):
        logits=beta*q;lp=logits-logsumexp(logits,axis=-1,keepdims=True);p=np.exp(lp)
        a=actions[:,t]
        if gradient:
            total-=lp[idx,a].sum()
            grad[0]+=beta*((p*dq).sum(-1)-dq[idx,a]).sum()
            grad[1]+=beta*((p*q).sum(-1)-q[idx,a]).sum()
        else:probs.append(p)
        update(a,rewards[:,t])
    return (total/actions.size,grad/actions.size) if gradient else np.stack(probs,axis=1)


def fit(actions,rewards,arms,cue):
    starts=[]
    for a in (.05,.3,.8):
        for b in (.5,5.,50.):
            f=minimize(lambda x:rw(x,actions,rewards,arms,cue,True),[a,np.log(b)],jac=True,
                       method='L-BFGS-B',bounds=[(0.,1.),(np.log(1e-4),np.log(1000.))],
                       options={'maxiter':300,'ftol':1e-12,'gtol':1e-8})
            starts.append(dict(theta=f.x.tolist(),nll=float(f.fun),success=bool(f.success),message=str(f.message),iterations=int(f.nit)))
    valid=[s for s in starts if s['success']]
    if not valid:raise RuntimeError('All RW starts failed')
    best=min(valid,key=lambda s:s['nll'])
    if min(s['nll'] for s in starts)<best['nll']-1e-6:raise RuntimeError('Unconverged start beats selected optimum')
    return dict(alpha=best['theta'][0],beta=float(np.exp(best['theta'][1])),theta=best['theta'],train_nll=best['nll'],starts=starts)


def metrics(p,q,op,oq,a):
    base=nll(p,a).mean(1);donor=nll(q,a[None]).mean((0,2))
    return dict(intact_nll=base,donor_nll=donor,delta_nll=donor-base,
                response_error=(.5*np.abs((q-p[None])-(oq-op[None])).sum(-1)).mean((0,2)))


def run(task,w):
    name=f'reward_w{round(w*100):03d}'
    if task=='restless':
        root=Path('outputs/weight_curve_a_20260915')
        with np.load(root/name/'train_000.npz') as z:ta=z['action'];tr=z['reward']
        with np.load(root/name/'val_000.npz') as z:d=dict(z)
        a=d['action'];r=d['reward'];cue=None;tcue=None;arms=4
        maps=np.load(root/name/'validation_assay.npz')['donor_mappings']
        scale=json.loads((root/'protocol.json').read_text())['q_scale']
        op=oracle_replay(d,r,scale)[:,150:]
        oq=np.array([oracle_replay(d,r[m],scale)[:,150:] for m in maps])
        select=lambda p:p[:,150:]
        evala=a[:,150:];participants=d['base_participant_id']
    else:
        root=Path('outputs/spatial_pilot_v1')
        def observable(split):
            rows=[json.loads(s) for s in (root/name/f'{split}_observable.jsonl').read_text().splitlines()]
            eps=[e for row in rows for e in row['rounds']]
            return (np.array([[x*5+y for x,y in e['choices']] for e in eps]),
                    np.array([e['rewards'] for e in eps]),
                    (np.array([e['initial_position'][0]*5+e['initial_position'][1] for e in eps]),np.array([e['initial_reward'] for e in eps])))
        ta,tr,tcue=observable('train');a,r,cue=observable('val');arms=25
        with np.load(Path('outputs/spatial_neural_analysis_v1')/(name+'_matched.npz')) as z:
            maps=z['donor_maps'];participants=z['participants'];op=z['oracle_B_intact'].reshape(50,40,25);oq=z['oracle_B_donor'].reshape(20,50,40,25)
            np.testing.assert_array_equal(a.reshape(50,8,20),z['actions'])
        select=lambda p:p.reshape(50,8,20,25)[:,6:].reshape(50,40,25)
        evala=a.reshape(50,8,20)[:,6:].reshape(50,40)
    fp=OUT/f'{task}_{name}_fit.json'
    if fp.exists():fitted=json.loads(fp.read_text())
    else:
        fitted=fit(ta,tr,arms,tcue);fp.write_text(json.dumps(fitted,indent=2))
    print('FIT',task,w,fitted['alpha'],fitted['beta'],fitted['train_nll'],flush=True)
    intact=rw(fitted['theta'],a,r,arms,cue);p=select(intact);q=[]
    for m in maps:
        changed=r[m] if task=='restless' else r.reshape(50,8,20)[m].reshape(400,20)
        altered=rw(fitted['theta'],a,changed,arms,cue)
        np.testing.assert_allclose(altered[:,0],intact[:,0],atol=1e-12)
        q.append(select(altered))
    q=np.array(q);vectors={}
    def add(model,v):
        for k,x in v.items():vectors[model+'_'+k]=x
    add('rw',metrics(p,q,op,oq,evala));add('oracle',metrics(op,oq,op,oq,evala))
    if task=='restless':
        with np.load(Path('outputs/reference_followup_fit_20260916')/(name+'_responses.npz')) as z:
            np.testing.assert_array_equal(z['participants'],participants)
            base=z['intact_nll'][:,0];delta=z['donor_delta_nll'][:,0]
            add('fit',dict(intact_nll=base,donor_nll=base+delta,delta_nll=delta,response_error=z['donor_response_error'][:,0]))
        for family in ('gru','transformer'):
            rows=[]
            for seed in (11,22,33):
                with np.load(Path('outputs/weight_neural_full_20260916')/f'{family}_{name}_full_seed{seed}'/'assay_full.npz') as z:
                    np.testing.assert_array_equal(z['participants'],participants);np.testing.assert_array_equal(z['donor_mappings'],maps)
                    np_=z['intact_probabilities'][:,150:].astype(float);nq=z['donor_probabilities'].astype(float)
                np_/=np_.sum(-1,keepdims=True);nq/=nq.sum(-1,keepdims=True)
                rows.append(metrics(np_,nq,op,oq,evala))
            add(family,{k:np.mean([row[k] for row in rows],axis=0) for k in rows[0]})
    else:
        root=Path('outputs/spatial_neural_analysis_v1')
        with np.load(root/(name+'_matched.npz')) as z:
            add('fit',metrics(z['fit_B_intact'].reshape(50,40,25),z['fit_B_donor'].reshape(20,50,40,25),op,oq,evala))
        for family in ('gru','transformer'):
            rows=[]
            for seed in (11,22,33):
                with np.load(root/f'{name}_{family}_full_seed{seed}_fixedshape.npz') as z:
                    np_=z['intact'][:,6:].reshape(50,40,25).astype(float);nq=z['donor'].reshape(20,50,40,25).astype(float)
                rows.append(metrics(np_,nq,op,oq,evala))
            add(family,{k:np.mean([row[k] for row in rows],axis=0) for k in rows[0]})
    for family in ('gru','transformer'):
        vectors[family+'_nll_minus_rw']=vectors[family+'_intact_nll']-vectors['rw_intact_nll']
        vectors[family+'_response_error_minus_rw']=vectors[family+'_response_error']-vectors['rw_response_error']
    np.savez_compressed(OUT/f'{task}_{name}.npz',participants=participants,donor_maps=maps,rw_intact=p,rw_donor=q,**vectors)
    return {k:summarize(v) for k,v in vectors.items()}


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles=[('oracle','Oracle','#444444'),('fit','Fitted generator','#b87826'),('rw','RW + softmax','#b14375'),('gru','GRU','#228b78'),('transformer','Transformer','#536eba')]
    for task in ('restless','spatial'):
        fig,axes=plt.subplots(1,4,figsize=(16,3.8))
        for ax,metric,title in zip(axes,('intact_nll','donor_nll','delta_nll','response_error'),('Original rewards','Donor-replaced rewards','Ablation effect','Response error vs oracle')):
            for key,label,color in styles:
                ss=[summary[task][str(w)][key+'_'+metric] for w in WEIGHTS]
                y=[s['mean'] for s in ss];ci=np.array([s['ci95'] for s in ss])
                ax.plot(WEIGHTS,y,'o-',label=label,color=color,ms=3)
                ax.fill_between(WEIGHTS,ci[:,0],ci[:,1],color=color,alpha=.10)
            ax.set(title=title,xlabel='Generating reward weight',ylabel='Probability-response error' if metric=='response_error' else 'Nats / choice')
            if metric in ('intact_nll','donor_nll'):ax.axhline(np.log(4 if task=='restless' else 25),color='#aaa',ls=':')
        lo=min(a.get_ylim()[0] for a in axes[:2]);hi=max(a.get_ylim()[1] for a in axes[:2])
        for ax in axes[:2]:ax.set_ylim(lo,hi)
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle(task.capitalize()+' task | Train-only pooled RW baseline',fontsize=12)
        fig.tight_layout();fig.savefig(OUT/f'{task}_comparison.png',dpi=160);fig.savefig(OUT/f'{task}_comparison.pdf');plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    protocol=dict(model='RW + softmax; alpha and beta only; no lapse, kernel, spatial generalization or GT weight',
        initialization='Raw Q0=50; rewards centered at50 and divided by100 in both tasks; beta uses these units',
        cue='Spatial initial free cue processed by the same RW update; reset each map',
        fit='Separate pooled train-only MLE per task/weight; 9 starts; L-BFGS-B analytical gradients',
        bounds=dict(alpha=[0,1],beta=[1e-4,1000]),
        selection='Minimum converged training NLL only; no ablation-based selection',
        evaluation='Existing validation B targets: restless trials151-200; spatial maps7-8',
        intervention='Existing20 donor derangements, cumulative chosen reward replacement; true choices fixed; cue retained',
        reference_caveat='Generator refits use within-participant calibration; RW and neural fits pooled on training participants',
        response_error='Mean over matched targets/donors of 0.5*L1[(p_donor-p_intact)-(oracle_donor-oracle_intact)]',
        uncertainty='Paired participant bootstrap conditional on fitted models/donor maps; neural metrics averaged across3 seeds',test_accessed=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary={}
    for task in ('spatial','restless'):
        summary[task]={}
        for w in WEIGHTS:
            summary[task][str(w)]=run(task,w)
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
            print('COMPLETE',task,w,flush=True)
    plot(summary);(OUT/'COMPLETE').write_text('14 pooled fits and matched validation assays complete.\n')


if __name__=='__main__':main()
