"""Small individual-level model/parameter recovery from observable data only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from mechcal.generators.spatial_mixture import SpatialConfig,coordinates,rbf

MODELS=('local','reward','mixture')


def prepare(record,config):
    rounds=record['rounds'];coords=coordinates(config.grid_size)
    actions=np.array([[r*config.grid_size+c for r,c in b['choices']] for b in rounds])
    cue=np.array([b['initial_position'][0]*config.grid_size+b['initial_position'][1] for b in rounds])
    rewards=(np.array([b['rewards'] for b in rounds])-config.reward_mean)/config.reward_sd
    cue_rewards=(np.array([b['initial_reward'] for b in rounds])-config.reward_mean)/config.reward_sd
    last=np.concatenate([cue[:,None],actions[:,:-1]],axis=1)
    distance=np.abs(coords[None,None,:,:]-coords[last][:,:,None,:]).sum(-1)
    return dict(actions=actions,rewards=rewards,cue=cue,cue_rewards=cue_rewards,distance=distance)


def decode(theta,model):
    if model=='local':return dict(gp_length=1.,reward_temperature=.2,local_temperature=float(np.exp(theta[0])),weight=0.)
    if model=='reward':return dict(gp_length=float(np.exp(theta[0])),reward_temperature=float(np.exp(theta[1])),local_temperature=.7,weight=1.)
    return dict(gp_length=float(np.exp(theta[0])),reward_temperature=float(np.exp(theta[1])),
                local_temperature=float(np.exp(theta[2])),weight=float(theta[3]))


def choice_probabilities(theta,model,data,config):
    p=decode(theta,model);a=data['actions'];rounds,trials=a.shape;n=config.grid_size**2
    local=softmax(-data['distance']/p['local_temperature'],axis=-1)
    chosen_local=np.take_along_axis(local,a[...,None],-1)[...,0]
    if model=='local':return (1-config.lapse)*chosen_local+config.lapse/n
    mean=np.zeros((rounds,n));cov=np.broadcast_to(rbf(coordinates(config.grid_size),p['gp_length']),(rounds,n,n)).copy()
    chosen_reward=np.empty_like(chosen_local)
    idx=np.arange(rounds);noise=(config.observation_sd/config.reward_sd)**2
    def update(arm,y):
        nonlocal mean,cov
        column=cov[idx,:,arm].copy();denom=cov[idx,arm,arm]+noise
        mean+=column/denom[:,None]*(y-mean[idx,arm])[:,None]
        cov-=column[:,:,None]*column[:,None,:]/denom[:,None,None]
        cov=(cov+cov.transpose(0,2,1))/2
    update(data['cue'],data['cue_rewards'])
    for t in range(trials):
        reward=softmax(mean/p['reward_temperature'],axis=-1)
        chosen_reward[:,t]=reward[idx,a[:,t]]
        update(a[:,t],data['rewards'][:,t])
    return (1-config.lapse)*(p['weight']*chosen_reward+(1-p['weight'])*chosen_local)+config.lapse/n


def fit_subject(task):
    w,record,config_dict,seed=task;c=SpatialConfig(**config_dict);all_data=prepare(record,c)
    train={k:v[:6] for k,v in all_data.items()};held={k:v[6:] for k,v in all_data.items()}
    rng=np.random.default_rng(seed);result=dict(weight=w,participant=record['participant'],models={})
    for model in MODELS:
        if model=='local':bounds=[(np.log(.15),np.log(3.))];starts=[np.log([x]) for x in (.3,.8,1.8)]
        elif model=='reward':
            bounds=[(np.log(.2),np.log(4.)),(np.log(.03),np.log(2.))]
            starts=[np.log([1.,.25])]+[np.array([rng.uniform(lo,hi) for lo,hi in bounds]) for _ in range(2)]
        else:
            bounds=[(np.log(.2),np.log(4.)),(np.log(.03),np.log(2.)),(np.log(.15),np.log(3.)),(0.,1.)]
            starts=[np.array([np.log(1.),np.log(.25),np.log(.8),x]) for x in (.15,.5,.85)]
        objective=lambda x:float(-np.log(choice_probabilities(x,model,train,c)).sum())
        fits=[minimize(objective,x,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=150,ftol=1e-10)) for x in starts]
        best=min(fits,key=lambda f:f.fun)
        train_nll=float(best.fun);held_nll=float(-np.log(choice_probabilities(best.x,model,held,c)).mean())
        result['models'][model]=dict(parameters=decode(best.x,model),train_nll=train_nll/120,
            heldout_nll=held_nll,bic=2*train_nll+len(bounds)*np.log(120),
            optimizer_success=bool(best.success),iterations=int(best.nit),
            start_objectives=[float(f.fun) for f in fits],message=str(best.message))
    result['winner_heldout']=min(MODELS,key=lambda m:result['models'][m]['heldout_nll'])
    result['winner_bic']=min(MODELS,key=lambda m:result['models'][m]['bic'])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subjects',type=int,default=10);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();root=Path('outputs/spatial_pilot_v1');out=Path('outputs/spatial_recovery_pilot_v1');out.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text());tasks=[];truth={}
    for wi,w in enumerate(protocol['weights']):
        folder=root/f'reward_w{round(w*100):03d}'
        records=[json.loads(s) for s in (folder/'train_observable.jsonl').read_text().splitlines()][:args.subjects]
        tasks += [(w,r,protocol['config'],31415+i) for i,r in enumerate(records)]
        # Truth is retained outside worker inputs and only used after fitting.
        with np.load(folder/'train_audit.npz') as z:
            for i,r in enumerate(records):truth[(w,r['participant'])]={k:float(z[k][i]) for k in ('gp_length','reward_temperature','local_temperature')}
    result=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(fit_subject,tasks):
            row['true_parameters']=truth[(row['weight'],row['participant'])];result.append(row)
            (out/'results.json').write_text(json.dumps(result,indent=2))
            print('FIT',len(result),len(tasks),row['weight'],row['participant'],row['winner_heldout'],flush=True)
    summary={}
    for w in protocol['weights']:
        rows=[r for r in result if r['weight']==w];est=np.array([r['models']['mixture']['parameters']['weight'] for r in rows])
        summary[str(w)]=dict(n=len(rows),weight_mean=float(est.mean()),weight_mae=float(np.abs(est-w).mean()),
            weight_sd=float(est.std()),heldout_winners={m:sum(r['winner_heldout']==m for r in rows) for m in MODELS},
            bic_winners={m:sum(r['winner_bic']==m for r in rows) for m in MODELS},
            heldout_nll={m:float(np.mean([r['models'][m]['heldout_nll'] for r in rows])) for m in MODELS})
    metadata=dict(summary=summary,subjects_per_weight=args.subjects,train_rounds=6,heldout_rounds=2,
                  starts=3,optimizer='L-BFGS-B, numerical gradients',lapse='fixed to known 0.1',
                  observation_noise='fixed task SD 2',optimizer_failures=sum(not r['models'][m]['optimizer_success'] for r in result for m in MODELS),
                  interpretation='Nested candidate models; endpoint ties and weak-component parameters are not identified. Pilot only.',
                  test_and_validation_used=False)
    (out/'summary.json').write_text(json.dumps(metadata,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for w in protocol['weights']:
        est=[r['models']['mixture']['parameters']['weight'] for r in result if r['weight']==w]
        axs[0].scatter(np.full(len(est),w),est,alpha=.5,s=18,color='#218575')
        axs[0].plot(w,np.mean(est),'ko',ms=5)
    axs[0].plot([0,1],[0,1],'k--',lw=1);axs[0].set(xlabel='Generating reward weight',ylabel='Fitted mixture weight',title='Individual parameter recovery',ylim=(-.03,1.03))
    for m,color in zip(MODELS,['#b97922','#527bb6','#218575']):
        axs[1].plot(protocol['weights'],[summary[str(w)]['heldout_nll'][m] for w in protocol['weights']],'-o',label=m,color=color)
    axs[1].set(xlabel='Generating reward weight',ylabel='Held-out choice NLL',title='Fit 6 maps; predict 2 new maps');axs[1].legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'recovery.png',dpi=180);fig.savefig(out/'recovery.pdf');plt.close(fig)
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
