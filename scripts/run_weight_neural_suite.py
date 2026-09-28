"""Train separate weight-specific models. Never load or evaluate the sealed test set."""
import argparse
import contextlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from mechcal.training.train_restless_transformer import train
from run_weight_curve_a import WEIGHTS

SOURCE=Path('outputs/weight_curve_a_20260915')


def prepare(root):
    for index,w in enumerate(WEIGHTS):
        name=f'reward_w{round(w*100):03d}'
        dest=root/name
        dest.mkdir(parents=True,exist_ok=True)
        for split in ('train','val'):
            output=dest/f'{split}_000.npz'
            if output.exists(): continue
            data=dict(np.load(SOURCE/name/f'{split}_000.npz'))
            data['condition_name']=np.full(len(data['action']),name)
            data['condition_index']=np.full(len(data['action']),index)
            np.savez_compressed(output,**data)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--families',nargs='+',default=['gru','transformer'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--input-modes',nargs='+',choices=['full','choice_only'],default=['full','choice_only'])
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data_root=args.output/'prepared_data'
    prepare(data_root)
    spec=dict(families=args.families,seeds=args.seeds,weights=list(WEIGHTS),
        input_modes=args.input_modes,test_evaluation_deferred=True,smoke=args.smoke,
        evaluation_plan='Matched suffix 151-200 intact, cumulative donor, previous-reward local, and suffix-only donor; no pooling with historical datasets',
        note='Validation used for development and early stopping. Test remains sealed pending frozen evaluation.')
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec: raise RuntimeError('Protocol mismatch')
    path.write_text(json.dumps(spec,indent=2))
    complete=[]
    weights=WEIGHTS[:1] if args.smoke else WEIGHTS
    for family in args.families:
        template=Path('configs/train/gru_full.yaml' if family=='gru' else 'configs/train/restless_transformer_eps10_large.yaml')
        for seed in args.seeds:
            for w in weights:
                condition=f'reward_w{round(w*100):03d}'
                for mode in args.input_modes:
                    name=f'{family}_{condition}_{mode}_seed{seed}'
                    run=args.output/name
                    if (run/'metrics.json').exists():
                        complete.append(name); continue
                    if run.exists(): raise RuntimeError(f'Inspect interrupted training before resume: {run}')
                    cfg=yaml.safe_load(template.read_text())
                    cfg.update(input_mode=mode,data_root=str(data_root/condition),defer_test_evaluation=True)
                    if args.smoke:
                        cfg.update(max_epochs=1,limits=dict(train=8,val=8),batch_size=8)
                    config=args.output/f'{name}.yaml'
                    config.write_text(yaml.safe_dump(cfg))
                    print('START',name,flush=True)
                    with (args.output/f'{name}.log').open('w') as handle,contextlib.redirect_stdout(handle):
                        metrics=train(config,run,condition,seed)
                    assert metrics['test'] is None and metrics['dataset_sizes']['test'] is None
                    complete.append(name)
                    (args.output/'progress.json').write_text(json.dumps(dict(completed=complete,last_completed=name),indent=2))
                    print('COMPLETE',name,'val',metrics['validation']['choice_nll'],flush=True)
    (args.output/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(runs=len(complete),test_evaluated=False)))


if __name__=='__main__': main()
