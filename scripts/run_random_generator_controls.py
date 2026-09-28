"""Independent fully initialized GRU/Transformer teachers; frozen, no fitting."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
import run_neural_generator_sweep as sweep

ROOT=Path('outputs/random_generator_controls_20260921')
FAMILY=None
ORIGINAL_GENERATE=sweep.generate

def generator(seed):
    torch.manual_seed(seed)
    path=Path('configs/train/gru_full.yaml' if FAMILY=='gru' else 'configs/train/restless_transformer_eps10_large.yaml')
    cfg=yaml.safe_load(path.read_text())['model']
    model=(sweep.CausalGRU(sweep.GRUConfig(**cfg)) if FAMILY=='gru' else sweep.CausalTransformer(sweep.TransformerConfig(**cfg)))
    for p in model.parameters():p.requires_grad_(False)
    folder=ROOT/FAMILY;folder.mkdir(parents=True,exist_ok=True)
    ck=folder/f'generator_seed{seed}.pt'
    if not ck.exists():torch.save(dict(model_state=model.state_dict(),model_config=cfg,seed=seed,family=FAMILY,initialization='all parameters fresh; no checkpoint loaded'),ck)
    else:
        old=torch.load(ck,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,old['model_state'][k]) for k,v in model.state_dict().items())
    return model.cuda().eval(),cfg

@torch.inference_mode()
def generate(model,seed,split,count,weight):
    if FAMILY=='gru':return ORIGINAL_GENERATE(model,seed,split,count,weight)
    cfg=sweep.RestlessConfig()
    env=np.array([int(np.random.SeedSequence([20260918,741,split,i]).generate_state(1)[0]) for i in range(count)])
    potential=np.array([sweep.generate_potential_rewards(cfg,sweep.generate_reward_schedule(cfg,int(s)),int(s)) for s in env])
    uniforms=np.random.default_rng(np.random.SeedSequence([20260918,seed,split,918])).random((count,200))
    actions=np.empty((count,200),int);rewards=np.empty_like(actions);probs=[]
    full=torch.zeros((count,401),dtype=torch.long,device='cuda');blind=full.clone()
    for t in range(200):
        parts=[]
        for start in range(0,count,32):
            stop=min(start+32,count)
            # The teacher sees only the real prefix, never a future reward.
            pf=model(full[start:stop,:2*t+1])[:,-1,1:5].float().softmax(-1)
            pb=model(blind[start:stop,:2*t+1])[:,-1,1:5].float().softmax(-1)
            parts.append((weight*pf+(1-weight)*pb).cpu().numpy())
        p=np.concatenate(parts).astype(float);p/=p.sum(-1,keepdims=True)
        a=(uniforms[:,t,None]>p.cumsum(-1)).sum(-1).clip(max=3)
        r=potential[np.arange(count),t,a];actions[:,t]=a;rewards[:,t]=r;probs.append(p)
        full[:,2*t+1]=torch.as_tensor(a+1,device='cuda');full[:,2*t+2]=torch.as_tensor(r+5,device='cuda')
        blind[:,2*t+1]=full[:,2*t+1];blind[:,2*t+2]=55
    tokens,mask=sweep.encode(actions,rewards)
    return dict(action=actions,reward=rewards,tokens=tokens,choice_target_mask=mask,oracle_probabilities=np.stack(probs,1),
      environment_seed=env,condition_index=np.zeros(count,int),condition_name=np.full(count,'neural_mixture'),kernel_sign=np.zeros(count,int),
      session_id=np.arange(count)+split*10000,base_participant_id=np.arange(count)+split*10000)

def smoke(family):
    global FAMILY
    FAMILY=family;result=[]
    for seed in (101,102,103):
        model,cfg=generator(seed)
        d=generate(model,seed,0,16,1.)
        a=d['action'];r=d['reward'];p=sweep.replay(model,a,r,1.)
        np.testing.assert_allclose(p,d['oracle_probabilities'],atol=5e-5,rtol=5e-5)
        donor=np.roll(r,1,axis=0);q=sweep.replay(model,a,donor,1.)
        np.testing.assert_allclose(p[:,0],q[:,0],atol=1e-7)
        np.testing.assert_allclose(sweep.replay(model,a,r,0.),sweep.replay(model,a,donor,0.),atol=1e-7)
        changed=r.copy();changed[:,100:]=0
        np.testing.assert_allclose(p[:,:101],sweep.replay(model,a,changed,1.)[:,:101],atol=5e-5,rtol=5e-5)
        row=dict(seed=seed,passed=True,entropy=float(-(p*np.log(p)).sum(-1).mean()),reward_tv=float((.5*abs(q-p).sum(-1)).mean()),
          replay_max_difference=float(abs(p-d['oracle_probabilities']).max()),choice_fractions=(np.bincount(a.ravel(),minlength=4)/a.size).tolist())
        result.append(row);print('SMOKE',family,row,flush=True)
        del model;torch.cuda.empty_cache()
    (ROOT/family/'smoke.json').write_text(json.dumps(result,indent=2))

def main():
    global FAMILY
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--family',choices=['gru','transformer','both'],default='both');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    families=('gru','transformer') if args.family=='both' else (args.family,)
    ROOT.mkdir(exist_ok=True)
    # Audit every prespecified teacher before spending time fitting any student.
    for family in families:smoke(family)
    if args.smoke_only:return
    for family in families:
        FAMILY=family;sweep.OUT=ROOT/family;sweep.SOURCE=Path(__file__)
        sweep.generator=generator;sweep.generate=generate
        sweep.PROTOCOL_EXTRA=dict(source_kind='initialization code, not pretrained checkpoint',generator_family=family,
          initialization='All parameters initialized from fixed seed; native model defaults; no checkpoint loaded; frozen FP32',
          policy='w * frozen random policy(real rewards) + (1-w) * same frozen policy(rewards 50); own closed-loop choices',
          limitations='One student seed; three generator seeds; weak reward sensitivity retained and disclosed; validation exploratory',
          selection='All 21 conditions retained; no entropy-based seed selection, temperature adjustment or posthoc rescaling')
        sys.argv=[sys.argv[0]];sweep.main()
    (ROOT/'COMPLETE').write_text('Requested families finished: '+','.join(families))

if __name__=='__main__':main()
