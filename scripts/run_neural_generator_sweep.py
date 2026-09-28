"""Paired neural-policy mixture sweep. No validation-based generator selection."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from run_neural_generator_pilot import (generator, encode, predict, SOURCE, HEADS,
    CausalGRU, GRUConfig, CausalTransformer, TransformerConfig, train,
    RestlessConfig, generate_reward_schedule, generate_potential_rewards)
from run_rw_baseline import fit, rw, metrics, WEIGHTS
from run_weight_curve_a import derangements
from ablate_spatial_references import nll

OUT = Path('outputs/neural_generator_sweep_v1')
PROTOCOL_EXTRA = {}

def replay(model, actions, rewards, weight):
    full = predict(model, encode(actions, rewards)[0])
    blind = predict(model, encode(actions, np.full_like(rewards, 50))[0])
    return weight * full + (1-weight) * blind

@torch.inference_mode()
def generate(model, seed, split, count, weight):
    cfg = RestlessConfig()
    env = np.array([int(np.random.SeedSequence([20260918,741,split,i]).generate_state(1)[0]) for i in range(count)])
    potential = np.array([generate_potential_rewards(cfg,generate_reward_schedule(cfg,int(s)),int(s)) for s in env])
    uniforms = np.random.default_rng(np.random.SeedSequence([20260918,seed,split,918])).random((count,200))
    bos = torch.zeros((count,1),dtype=torch.long,device='cuda')
    sf,hf = model.rnn(model.embedding(bos)); sb,hb = model.rnn(model.embedding(bos))
    actions=np.empty((count,200),int); rewards=np.empty_like(actions); probs=[]
    for t in range(200):
        pf=model.head(sf[:,-1])[:,1:5].softmax(-1)
        pb=model.head(sb[:,-1])[:,1:5].softmax(-1)
        p=(weight*pf+(1-weight)*pb).cpu().numpy().astype(float);p/=p.sum(-1,keepdims=True)
        a=(uniforms[:,t,None]>p.cumsum(-1)).sum(-1).clip(max=3)
        r=potential[np.arange(count),t,a];actions[:,t]=a;rewards[:,t]=r;probs.append(p)
        xf=torch.as_tensor(np.stack([a+1,r+5],1),device='cuda')
        xb=xf.clone();xb[:,1]=55
        sf,hf=model.rnn(model.embedding(xf),hf);sb,hb=model.rnn(model.embedding(xb),hb)
    tokens,mask=encode(actions,rewards)
    return dict(action=actions,reward=rewards,tokens=tokens,choice_target_mask=mask,
        oracle_probabilities=np.stack(probs,1),environment_seed=env,
        condition_index=np.zeros(count,int),condition_name=np.full(count,'neural_mixture'),
        kernel_sign=np.zeros(count,int),session_id=np.arange(count)+split*10000,
        base_participant_id=np.arange(count)+split*10000)

@torch.inference_mode()
def student_predict(model,tokens,mode):
    if mode=='full':return predict(model,tokens)
    tokens=np.concatenate([tokens[:,:1],tokens[:,1::2]],1);result=[]
    for start in range(0,len(tokens),32):
        size=min(32,len(tokens)-start)
        x=torch.zeros((32,201),dtype=torch.long,device='cuda')
        x[:size]=torch.as_tensor(tokens[start:start+size],device='cuda')
        result.append(model(x[:,:-1])[:size,:,1:5].float().softmax(-1).cpu().numpy())
    return np.concatenate(result).astype(float)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    protocol=dict(weights=WEIGHTS,heads=list(HEADS),train=500,val=100,trials=200,
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        policy='w * frozen neural policy(real rewards) + (1-w) * same frozen neural policy(all rewards 50); separate persistent recurrent states',
        pairing='Same environment seeds and action uniforms across weights; disjoint train/val environments',
        training='Fresh pooled GRU/Transformer, full and choice_only, seed11, existing early stopping; RW train-only pooled 9-start',
        evaluation='A all200 trials; B trials151-200,20 paired donors, actual choices fixed; no test access',
        limitations='Pretrained symbolic-data torso; constant50 branch is reward-blind not a learned optimal choice-only oracle; one student seed; validation exploratory',
        selection='All21 generators retained irrespective of entropy or outcomes')
    protocol.update(PROTOCOL_EXTRA)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    for seed in HEADS:
        for weight in WEIGHTS:
            root=OUT/f'head{seed}_w{round(weight*100):03d}';root.mkdir(exist_ok=True)
            teacher,cfg=generator(seed);data={}
            for name,count,split in [('train',500,0),('val',100,1)]:
                path=root/f'{name}_000.npz'
                if path.exists():
                    with np.load(path) as z:data[name]=dict(z)
                else:
                    data[name]=generate(teacher,seed,split,count,weight);np.savez_compressed(path,**data[name])
            assert not set(data['train']['environment_seed'])&set(data['val']['environment_seed'])
            d=data['val'];a=d['action'];r=d['reward'];maps=derangements(len(a),20,913)
            p=replay(teacher,a,r,weight)
            np.testing.assert_allclose(p,d['oracle_probabilities'],atol=5e-5,rtol=5e-5)
            q=replay(teacher,a,r[maps[0]],weight)
            np.testing.assert_allclose(p[:,0],q[:,0],atol=1e-7)
            if weight==0:np.testing.assert_allclose(p,q,atol=1e-7)
            # Both branches have identical initial state; changing future rewards cannot affect earlier predictions.
            changed=r.copy();changed[:,100:]=0
            np.testing.assert_allclose(p[:,:101],replay(teacher,a,changed,weight)[:,:101],atol=5e-5,rtol=5e-5)
            print('AUDIT_PASS',seed,weight,flush=True)
            if args.audit_only:
                del teacher;torch.cuda.empty_cache();continue
            ref=root/'reference.npz'
            if not ref.exists():
                oq=np.array([replay(teacher,a,r[m],weight)[:,150:] for m in maps])
                np.savez_compressed(ref,oracle_intact=p,oracle_donor=oq,donor_maps=maps)
            with np.load(ref) as z:oq=z['oracle_donor']
            del teacher;torch.cuda.empty_cache()
            result={'oracle_A':float(nll(p,a).mean())}
            fp=root/'rw_fit.json'
            if not fp.exists():fp.write_text(json.dumps(fit(data['train']['action'],data['train']['reward'],4,None),indent=2))
            theta=json.loads(fp.read_text())['theta'];rp=rw(theta,a,r,4)
            rq=np.array([rw(theta,a,r[m],4)[:,150:] for m in maps])
            result['rw_A']=float(nll(rp,a).mean())
            vectors={f'rw_{k}':v for k,v in metrics(rp[:,150:],rq,p[:,150:],oq,a[:,150:]).items()}
            for family in ('gru','transformer'):
                for mode in ('full','choice_only'):
                    run=root/f'{family}_{mode}_seed11'
                    if not (run/'metrics.json').exists():
                        config=yaml.safe_load(Path('configs/train/gru_full.yaml' if family=='gru' else 'configs/train/restless_transformer_eps10_large.yaml').read_text())
                        config.update(data_root=str(root),defer_test_evaluation=True,input_mode=mode,condition='neural_mixture')
                        cp=root/f'{family}_{mode}.yaml';cp.write_text(yaml.safe_dump(config))
                        print('TRAIN',seed,weight,family,mode,flush=True)
                        with (root/f'{family}_{mode}.log').open('w') as f,contextlib.redirect_stdout(f):train(cp,run,'neural_mixture',11)
                    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
                    model=CausalGRU(GRUConfig(**ck['model_config'])) if family=='gru' else CausalTransformer(TransformerConfig(**ck['model_config']))
                    model.load_state_dict(ck['model_state']);model=model.cuda().eval()
                    sp=student_predict(model,d['tokens'],mode)
                    result[f'{family}_{mode}_A']=float(nll(sp,a).mean())
                    payload=dict(intact=sp)
                    if mode=='full':
                        sq=np.array([predict(model,encode(a,r[m])[0])[:,150:] for m in maps])
                        payload['donor']=sq
                        vectors.update({f'{family}_{k}':v for k,v in metrics(sp[:,150:],sq,p[:,150:],oq,a[:,150:]).items()})
                    np.savez_compressed(root/f'{family}_{mode}_assay.npz',**payload)
                    del model;torch.cuda.empty_cache()
            vectors.update({f'oracle_{k}':v for k,v in metrics(p[:,150:],oq,p[:,150:],oq,a[:,150:]).items()})
            np.savez_compressed(root/'B_vectors.npz',**vectors)
            (root/'summary.json').write_text(json.dumps(result,indent=2))
            print('CELL_COMPLETE',seed,weight,result,flush=True)
    (OUT/('AUDIT_COMPLETE' if args.audit_only else 'COMPLETE')).write_text('All21 cells completed.\n')
    if not args.audit_only:
        import plot_neural_generator_sweep as plotting
        plotting.ROOT=OUT
        plotting.plot()

if __name__=='__main__':main()
