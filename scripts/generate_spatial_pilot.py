"""Generate a paired seven-weight pilot, with separate observable/audit files."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from mechcal.generators.spatial_mixture import SpatialConfig, simulate_subject, observable_record, coordinates

WEIGHTS = [0., .1, .3, .5, .7, .9, 1.]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('outputs/spatial_pilot_v1'))
    parser.add_argument('--train',type=int,default=200)
    parser.add_argument('--val',type=int,default=50)
    parser.add_argument('--test',type=int,default=50)
    parser.add_argument('--seed',type=int,default=20260918)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output directory; never overwrite a pilot')
    args.output.mkdir(parents=True)
    config=SpatialConfig()
    protocol=dict(config=asdict(config),weights=WEIGHTS,counts={s:getattr(args,s) for s in ('train','val','test')},
                  master_seed=args.seed,task='5x5 stationary spatial bandit, 8 independent rounds of 20 choices',
                  mixture='(1-lapse)*(w*softmax(GP_mean/tau_R)+(1-w)*softmax(-Manhattan_distance/tau_L))+lapse/25',
                  parameters=dict(gp_length=[.8,1.6],reward_temperature=[.12,.35],local_temperature=[.5,1.1]),
                  parameter_distribution='Independent uniform per participant; paired across weights',
                  rewards='Unbounded Gaussian landscape: mean 50, amplitude SD 15; Gaussian observation SD 2; no clipping/rounding',
                  information='Only cue and chosen rewards observable; no UCB, latent map and parameters audit-only',
                  split='Distinct participants and environment seeds across splits; test generated but not summarized',
                  pairing='Same environments, initial cues, potential-outcome noise, and choice uniforms across weights',
                  reference='Wu et al. 2018, doi:10.1038/s41562-018-0467-4; adapted task, not a replication',
                  status='Pilot generation only; not a locked confirmatory dataset or neural training run')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summaries={}
    coords=coordinates(config.grid_size)
    for split_idx,split in enumerate(('train','val','test')):
        n=getattr(args,split)
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,split_idx,999]))
        parameters=[dict(gp_length=float(rng.uniform(.8,1.6)),reward_temperature=float(rng.uniform(.12,.35)),
                         local_temperature=float(rng.uniform(.5,1.1))) for _ in range(n)]
        seeds=[int(np.random.SeedSequence([args.seed,split_idx,i,888]).generate_state(1)[0]) for i in range(n)]
        assert len(set(seeds))==n
        for w in WEIGHTS:
            folder=args.output/f'reward_w{round(w*100):03d}'
            folder.mkdir(exist_ok=True)
            episodes=[]
            with (folder/f'{split}_observable.jsonl').open('w',encoding='utf-8') as f:
                for i in range(n):
                    e=simulate_subject(config,parameters[i],w,seeds[i]);episodes.append(e)
                    f.write(json.dumps(observable_record(config,f'{split}_{i:04d}',e))+'\n')
            arrays={k:np.stack([e[k] for e in episodes]) for k in episodes[0]}
            arrays.update(participant=np.array([f'{split}_{i:04d}' for i in range(n)]),
                          subject_seed=np.array(seeds,dtype=np.uint32),
                          **{k:np.array([p[k] for p in parameters]) for k in parameters[0]})
            np.savez_compressed(folder/f'{split}_audit.npz',**arrays)
            if split!='test':
                a=arrays['actions'];p=arrays['probabilities']
                last=np.concatenate([arrays['cue_arm'][...,None],a[...,:-1]],axis=-1)
                nll=-np.log(np.take_along_axis(p,a[...,None],axis=-1)[...,0])
                summaries.setdefault(split,{})[str(w)]=dict(oracle_nll=float(nll.mean()),
                    mean_reward=float(arrays['rewards'].mean()),stay_rate=float((a==last).mean()),
                    mean_manhattan_step=float(np.abs(coords[a]-coords[last]).sum(-1).mean()),
                    unique_arms_per_round=float(np.mean([len(np.unique(row)) for row in a.reshape(-1,config.trials)])),
                    oracle_entropy=float((-p*np.log(p)).sum(-1).mean()))
            print('GENERATED',split,w,n,flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))
    files=sorted(p for p in args.output.rglob('*') if p.is_file())
    manifest={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
