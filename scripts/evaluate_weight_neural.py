"""Matched validation assays for completed seven-weight full-input models."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.models.gru import CausalGRU, GRUConfig
from run_reference_followup import ROOT, load, groups
from run_weight_curve_a import replay, nll

@torch.no_grad()
def predict(model,tokens,target=None):
    result=[]
    stop=400 if target is None else 2*target+1
    for batch in tokens.split(32):
        logits=model(batch[:,:stop].cuda()).float()
        logits=logits[:,0::2,1:5] if target is None else logits[:,-1:,1:5]
        result.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('outputs/weight_neural_full_20260916'))
    ap.add_argument('--run');ap.add_argument('--skip-local',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4)
    scale=json.loads((ROOT/'protocol.json').read_text())['q_scale']
    runs=[args.root/args.run] if args.run else sorted(p.parent for p in args.root.glob('*/metrics.json'))
    for run in runs:
        if not (run/'metrics.json').exists():continue
        target=run/('assay_cumulative.npz' if args.skip_local else 'assay_full.npz')
        if target.exists():continue
        family,_,wcode,mode,seed=run.name.split('_');assert mode=='full'
        w=int(wcode[1:])/100;d=load(w);tokens=torch.tensor(d['tokens'].astype(np.int64))
        maps=np.load(ROOT/f'reward_w{int(w*100):03d}'/'validation_assay.npz')['donor_mappings']
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        model=(CausalGRU(GRUConfig(**ck['model_config'])) if family=='gru' else CausalTransformer(TransformerConfig(**ck['model_config']))).cuda().eval()
        model.load_state_dict(ck['model_state'])
        ip=predict(model,tokens);op=replay(d,d['reward'],scale)
        base=nll(ip,d['action']);ob=nll(op,d['action'])
        metrics=dict(intact_nll=base[:,150:].mean(1));arrays=dict(participants=d['base_participant_id'],donor_mappings=maps,intact_probabilities=ip)
        for operation in ('donor','suffix_donor','local'):
            if args.skip_local and operation=='local':continue
            losses=[];errors=[];prob=[]
            for mapping in maps:
                changed=tokens.clone();r=d['reward'][mapping].copy()
                if operation=='local':
                    parts=[]
                    for t in range(150,200):
                        changed=tokens.clone();changed[:,2*t]=tokens[mapping,2*t]
                        parts.append(predict(model,changed,t))
                    pp=np.concatenate(parts,axis=1)
                else:
                    start=2 if operation=='donor' else 302
                    changed[:,start::2]=tokens[mapping,start::2]
                    if operation=='suffix_donor':r[:,:150]=d['reward'][:,:150]
                    pp=predict(model,changed)[:,150:]
                oo=replay(d,r,scale,operation=='local')[:,150:]
                losses.append(nll(pp,d['action'][:,150:])-base[:,150:])
                errors.append(.5*np.abs((pp-ip[:,150:])-(oo-op[:,150:])).sum(-1))
                prob.append(pp)
            metrics[operation+'_delta_nll']=np.mean(losses,axis=(0,2))
            metrics[operation+'_response_error']=np.mean(errors,axis=(0,2))
            arrays[operation+'_probabilities']=np.stack(prob)
        arrays.update(metrics)
        np.savez_compressed(target,**arrays)
        target.with_suffix('.json').write_text(json.dumps(dict(split='val',targets='151-200',precision='fp32',groups=groups(metrics,d['kernel_sign'])),indent=2))
        print('EVALUATED',run.name,flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
