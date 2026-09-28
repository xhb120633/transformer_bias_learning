"""Frozen trained GRU torso + independently reset head; closed-loop pilot."""
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from mechcal.models.gru import CausalGRU,GRUConfig
from mechcal.models.causal_transformer import CausalTransformer,TransformerConfig
from mechcal.generators.four_armed_restless import RestlessConfig,generate_reward_schedule,generate_potential_rewards,tokenize_session
from mechcal.training.train_restless_transformer import train
from run_rw_baseline import fit as fit_rw,rw,metrics
from run_weight_curve_a import derangements
from ablate_spatial_references import summarize,nll

OUT=Path('outputs/neural_generator_pilot_v1')
SOURCE=Path('outputs/weight_neural_full_20260916/gru_reward_w100_full_seed11/best.pt')
HEADS=(101,102,103)


@torch.inference_mode()
def predict(model,tokens):
    result=[]
    for start in range(0,len(tokens),32):
        x=torch.zeros((32,401),dtype=torch.long,device='cuda')
        size=min(32,len(tokens)-start)
        x[:size]=torch.as_tensor(tokens[start:start+size],device='cuda')
        result.append(model(x[:,:-1])[:size,::2,1:5].float().softmax(-1).cpu().numpy())
    return np.concatenate(result).astype(float)


def encode(actions,rewards):
    rows=[tokenize_session(a,r) for a,r in zip(actions,rewards)]
    return np.stack([x[0] for x in rows]),np.stack([x[1] for x in rows])


def generator(seed):
    ck=torch.load(SOURCE,map_location='cpu',weights_only=True)
    model=CausalGRU(GRUConfig(**ck['model_config']));model.load_state_dict(ck['model_state'])
    torch.manual_seed(seed);model.head.reset_parameters()
    for name,value in model.state_dict().items():
        if not name.startswith('head.'):assert torch.equal(value,ck['model_state'][name])
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model.cuda().eval(),ck['model_config']


@torch.inference_mode()
def generate(model,seed,split,count):
    config=RestlessConfig();potential=[];envseeds=[]
    for i in range(count):
        es=int(np.random.SeedSequence([20260918,741,split,i]).generate_state(1)[0]);envseeds.append(es)
        potential.append(generate_potential_rewards(config,generate_reward_schedule(config,es),es))
    potential=np.array(potential);uniforms=np.random.default_rng(np.random.SeedSequence([20260918,seed,split,918])).random((count,200))
    state,hidden=model.rnn(model.embedding(torch.zeros((count,1),dtype=torch.long,device='cuda')))
    actions=np.empty((count,200),int);rewards=np.empty_like(actions);prob=[]
    for t in range(200):
        p=model.head(state[:,-1])[:,1:5].softmax(-1).cpu().numpy().astype(float);p/=p.sum(-1,keepdims=True)
        a=(uniforms[:,t,None]>p.cumsum(-1)).sum(-1).clip(max=3)
        r=potential[np.arange(count),t,a];actions[:,t]=a;rewards[:,t]=r;prob.append(p)
        x=torch.as_tensor(np.stack([a+1,r+5],axis=1),device='cuda')
        state,hidden=model.rnn(model.embedding(x),hidden)
    tokens,mask=encode(actions,rewards)
    return dict(action=actions,reward=rewards,tokens=tokens,choice_target_mask=mask,
                oracle_probabilities=np.stack(prob,axis=1),environment_seed=np.array(envseeds),
                condition_index=np.zeros(count,int),condition_name=np.full(count,f'head{seed}'),
                kernel_sign=np.zeros(count,int),session_id=np.arange(count)+split*10000,
                base_participant_id=np.arange(count)+split*10000)


def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    protocol=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        change='Freeze source GRU embedding/recurrent layers; PyTorch default reset of full output head only',
        heads=list(HEADS),policy='Conditional softmax over4 choices, temperature1, no added lapse',
        data='Own sampled choices and actual chosen rewards in closed loop; no old subject trajectories',
        train=500,val=100,trials=200,test_generated=False,
        gate='Training-only: mean entropy<ln4-.01; max marginal choice fraction<.90; mean donor probability TV>.01',
        gate_note='Keep all diagnostics; no selection based on student performance or RW winning; weak heads stop before fitting',
        students='Fresh RW/GRU/Transformer full-input only, separate perhead, one training seed11 pilot; no shared teacher weights',
        evaluation='Val trials151-200;20 donor derangements; real choices fixed; train-only RW MLE; no teacher parameters used in fitting',
        limitation='Teacher torso inherits representations learned from original symbolic-generated data; not an independently learned neural mechanism')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));summary={}
    for seed in HEADS:
        root=OUT/f'head{seed}';root.mkdir(exist_ok=True)
        teacher,cfg=generator(seed)
        torch.save(dict(model_state=teacher.state_dict(),model_config=cfg,head_seed=seed),root/'generator.pt')
        data={}
        for split,count,idx in [('train',500,0),('val',100,1)]:
            path=root/f'{split}_000.npz'
            if path.exists():
                with np.load(path) as z:data[split]=dict(z)
            else:
                data[split]=generate(teacher,seed,idx,count);np.savez_compressed(path,**data[split])
        assert not set(data['train']['environment_seed'])&set(data['val']['environment_seed'])
        d=data['train'];p=predict(teacher,d['tokens'])
        np.testing.assert_allclose(p,d['oracle_probabilities'],atol=5e-5,rtol=5e-5)
        mapping=derangements(len(p),1,951)[0]
        dt,_=encode(d['action'],d['reward'][mapping]);q=predict(teacher,dt)
        entropy=float(-(p*np.log(p)).sum(-1).mean());tv=float((.5*np.abs(q-p).sum(-1))[:,1:].mean())
        marginal=np.bincount(d['action'].ravel(),minlength=4)/d['action'].size
        gate=dict(entropy=entropy,max_marginal=float(marginal.max()),choice_fractions=marginal.tolist(),reward_donor_tv=tv,
                  passed=bool(entropy<np.log(4)-.01 and marginal.max()<.90 and tv>.01))
        (root/'gate.json').write_text(json.dumps(gate,indent=2));print('GATE',seed,gate,flush=True)
        if not gate['passed']:
            summary[str(seed)]={'gate':gate};del teacher;torch.cuda.empty_cache();continue
        d=data['val'];a=d['action'][:,150:];maps=derangements(100,20,913)
        op=predict(teacher,d['tokens'])[:,150:]
        oq=np.array([predict(teacher,encode(d['action'],d['reward'][m])[0])[:,150:] for m in maps])
        del teacher;torch.cuda.empty_cache()
        train_data=data['train'];fitpath=root/'rw_fit.json'
        if fitpath.exists():fitted=json.loads(fitpath.read_text())
        else:
            fitted=fit_rw(train_data['action'],train_data['reward'],4,None);fitpath.write_text(json.dumps(fitted,indent=2))
        rp=rw(fitted['theta'],d['action'],d['reward'],4)[:,150:]
        rq=np.array([rw(fitted['theta'],d['action'],d['reward'][m],4)[:,150:] for m in maps])
        vectors={}
        def add(label,values):
            for k,v in values.items():vectors[label+'_'+k]=v
        add('oracle',metrics(op,oq,op,oq,a));add('rw',metrics(rp,rq,op,oq,a))
        for family in ('gru','transformer'):
            run=root/f'{family}_seed11'
            if not (run/'metrics.json').exists():
                config=yaml.safe_load(Path('configs/train/gru_full.yaml' if family=='gru' else 'configs/train/restless_transformer_eps10_large.yaml').read_text())
                config.update(data_root=str(root),defer_test_evaluation=True,input_mode='full',condition=f'head{seed}')
                cp=root/f'{family}.yaml';cp.write_text(yaml.safe_dump(config))
                print('TRAIN',seed,family,flush=True)
                with (root/f'{family}.log').open('w') as f,contextlib.redirect_stdout(f):train(cp,run,f'head{seed}',11)
            ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            model=CausalGRU(GRUConfig(**ck['model_config'])) if family=='gru' else CausalTransformer(TransformerConfig(**ck['model_config']))
            model.load_state_dict(ck['model_state']);model=model.cuda().eval()
            p=predict(model,d['tokens'])[:,150:]
            q=np.array([predict(model,encode(d['action'],d['reward'][m])[0])[:,150:] for m in maps])
            add(family,metrics(p,q,op,oq,a));np.savez_compressed(root/f'{family}_assay.npz',intact=p,donor=q)
            del model;torch.cuda.empty_cache()
        np.savez_compressed(root/'assay.npz',donor_maps=maps,actions=a,oracle_intact=op,oracle_donor=oq,**vectors)
        summary[str(seed)]={'gate':gate,'metrics':{k:summarize(v) for k,v in vectors.items()}}
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('HEAD_COMPLETE',seed,flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'COMPLETE').write_text('All three prespecified heads assessed; see gates for trained heads.\n')


if __name__=='__main__':main()
