"""Development-only common-history controls and prefix-fitted symbolic assays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from run_weight_curve_a import WEIGHTS, replay, nll, estimate

ROOT = Path('outputs/weight_curve_a_20260915')


def load(w):
    return dict(np.load(ROOT / f'reward_w{round(w*100):03d}' / 'val_000.npz'))


def params(d, alpha, w, sign):
    return dict(d, alpha=np.asarray(alpha), beta_reward=np.asarray(w),
                beta_kernel=np.asarray(sign)*(1-np.asarray(w)))


def groups(metrics, signs):
    return {g: {k: estimate(x, mask) for k, x in metrics.items()}
            for g, mask in dict(all=np.ones(len(signs), bool), positive=signs>0, negative=signs<0).items()}


def common_history(out, scale):
    # Every target weight is replayed on every source-weight history.
    summary = {}
    vectors = {}
    for w in WEIGHTS:
        collected = {k: [] for k in ('donor_tv', 'donor_expected_delta_nll', 'local_tv', 'local_expected_delta_nll')}
        for source in WEIGHTS:
            d = load(source)
            maps = np.load(ROOT/f'reward_w{round(source*100):03d}'/'validation_assay.npz')['donor_mappings']
            changed = params(d, d['alpha'], np.full(len(d['alpha']), w), d['kernel_sign'])
            p = replay(changed, d['reward'], scale)
            for local, name in ((False, 'donor'), (True, 'local')):
                tv, kl = [], []
                for m in maps:
                    pp = replay(changed, d['reward'][m], scale, local)
                    tv.append((.5*np.abs(pp-p).sum(-1))[:,1:].mean(1))
                    kl.append((p*np.log(p/pp)).sum(-1)[:,1:].mean(1))
                collected[name+'_tv'].append(np.mean(tv, axis=0))
                collected[name+'_expected_delta_nll'].append(np.mean(kl, axis=0))
        # Preserve base-participant clusters across all seven source histories.
        metrics = {k: np.mean(v, axis=0) for k,v in collected.items()}
        summary[str(w)] = groups(metrics, d['kernel_sign'])
        vectors.update({f'w{w}_{k}': v for k,v in metrics.items()})
        print('COMMON_HISTORY', w, flush=True)
    (out/'common_history.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out/'common_history_vectors.npz', **vectors)


def grid_fit(actions, rewards, scale, resolution=41):
    """Only a supplied prefix enters estimation. No true covariates/weight bounds."""
    aa = np.unique(np.r_[np.linspace(.01,.99,resolution), .25])
    ww = np.linspace(0,1,51 if resolution == 41 else 101)
    alpha, weight, sign = np.meshgrid(aa,ww,[-1.,1.], indexing='ij')
    candidates = np.stack([alpha.ravel(),weight.ravel(),sign.ravel()],1)
    agrid = torch.tensor(aa, device='cuda', dtype=torch.float64)
    wg = torch.tensor(ww, device='cuda', dtype=torch.float64)
    sg = torch.tensor([-1.,1.], device='cuda', dtype=torch.float64)
    all_loss = []
    for start in range(0,len(actions),16):
        ac = torch.tensor(actions[start:start+16], device='cuda', dtype=torch.long)
        rw = torch.tensor(rewards[start:start+16], device='cuda', dtype=torch.float64)
        n = len(ac)
        q = torch.full((n,len(aa),4),50., device='cuda', dtype=torch.float64)
        loss = torch.zeros((n,len(aa),len(ww),2),device='cuda',dtype=torch.float64)
        rows = torch.arange(n,device='cuda')[:,None]
        ais = torch.arange(len(aa),device='cuda')[None,:]
        for t in range(ac.shape[1]):
            z = (q-q.mean(-1,keepdim=True))/scale
            logits = z[:,:,None,None,:]*wg[None,None,:,None,None]
            logits = logits.expand(n,len(aa),len(ww),2,4).clone()
            if t:
                kernel = torch.nn.functional.one_hot(ac[:,t-1],4).double()
                logits += kernel[:,None,None,None,:]*(1-wg)[None,None,:,None,None]*sg[None,None,None,:,None]
            win = torch.isclose(logits,logits.amax(-1,keepdim=True),atol=1e-12,rtol=0)
            ix = ac[:,t,None,None,None,None].expand(n,len(aa),len(ww),2,1)
            chosen = win.gather(-1,ix).squeeze(-1)
            loss -= torch.log(.025+.9*chosen/win.sum(-1))
            choice = ac[:,t,None]
            old = q[rows,ais,choice]
            q[rows,ais,choice] = old+agrid[None,:]*(rw[:,t,None]-old)
        all_loss.append(loss.flatten(1).cpu().numpy())
    losses = np.concatenate(all_loss)
    selected, near_ranges, tied_counts, near_counts, fit_losses = [], [], [], [], []
    for row in losses:
        best = int(row.argmin())
        near = np.flatnonzero(row <= row[best]+1.+1e-10)
        # Report full parameter ranges, but only a labelled finite response sample.
        wanted = [best]
        for col in range(3):
            wanted += [int(near[candidates[near,col].argmin()]), int(near[candidates[near,col].argmax()])]
        wanted += near[np.linspace(0,len(near)-1,8).astype(int)].tolist()
        unique = list(dict.fromkeys(wanted))[:8]
        unique += [best]*(8-len(unique))
        selected.append(candidates[unique])
        near_ranges.append(np.stack([candidates[near].min(0), candidates[near].max(0)]))
        tied_counts.append(np.sum(np.isclose(row,row[best],atol=1e-10,rtol=0)))
        near_counts.append(len(near))
        fit_losses.append(row[unique])
    return dict(parameters=np.asarray(selected), ranges=np.asarray(near_ranges),
                tied_counts=np.asarray(tied_counts), near_counts=np.asarray(near_counts),
                fit_total_nll=np.asarray(fit_losses), candidate_grid=candidates)


def fit_assay(d, fits, scale, maps):
    original = replay(d,d['reward'],scale)
    truth_loss = nll(original,d['action'])
    per = {k: [] for k in ('intact_nll','intact_tv','donor_delta_nll','donor_response_error',
                           'local_delta_nll','local_response_error','suffix_donor_delta_nll','suffix_donor_response_error')}
    oracle_metrics = {}
    oracle_changed = {}
    for mode in ('donor','local','suffix_donor'):
        pp=[]
        for m in maps:
            r=d['reward'][m].copy()
            if mode=='suffix_donor': r[:,:150]=d['reward'][:,:150]
            pp.append(replay(d,r,scale,mode=='local'))
        oracle_changed[mode]=np.stack(pp)
        dl = np.stack([nll(p,d['action'])-truth_loss for p in pp])
        oracle_metrics[mode+'_delta_nll']=dl[:,:,150:].mean((0,2))
    for k in range(8):
        pa=fits['parameters'][:,k]
        fitted=params(d,pa[:,0],pa[:,1],pa[:,2])
        intact=replay(fitted,d['reward'],scale)
        base=nll(intact,d['action'])
        if k==0:
            assert np.allclose(base[:,:150].sum(1),fits['fit_total_nll'][:,0],atol=1e-7)
        per['intact_nll'].append(base[:,150:].mean(1))
        per['intact_tv'].append((.5*np.abs(intact-original).sum(-1))[:,150:].mean(1))
        for mode in ('donor','local','suffix_donor'):
            delta, error=[],[]
            for rep,m in enumerate(maps):
                r=d['reward'][m].copy()
                if mode=='suffix_donor': r[:,:150]=d['reward'][:,:150]
                perturbed=replay(fitted,r,scale,mode=='local')
                delta.append((nll(perturbed,d['action'])-base)[:,150:].mean(1))
                response=(perturbed-intact)-(oracle_changed[mode][rep]-original)
                error.append((.5*np.abs(response).sum(-1))[:,150:].mean(1))
            per[mode+'_delta_nll'].append(np.mean(delta,axis=0))
            per[mode+'_response_error'].append(np.mean(error,axis=0))
    per={k: np.stack(v,axis=1) for k,v in per.items()}
    metrics={k:v[:,0] for k,v in per.items()}
    metrics.update({'oracle_'+k:v for k,v in oracle_metrics.items()})
    metrics['oracle_intact_nll']=truth_loss[:,150:].mean(1)
    metrics['alpha_abs_error']=np.abs(fits['parameters'][:,0,0]-d['alpha'])
    metrics['weight_abs_error']=np.abs(fits['parameters'][:,0,1]-(1-d['lambda_choice']))
    for mode in ('donor','local','suffix_donor'):
        metrics[mode+'_sampled_near_optimal_effect_span']=np.ptp(per[mode+'_delta_nll'],axis=1)
    return groups(metrics,d['kernel_sign']), dict(**per,**{'oracle_'+k:v for k,v in oracle_metrics.items()},
        oracle_intact_nll=truth_loss[:,150:].mean(1),participants=d['base_participant_id'],kernel_sign=d['kernel_sign'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['common','fit'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--resolution',type=int,default=41)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    scale=protocol['q_scale']
    spec=dict(phase=args.phase,split='val',fit_trials=150,score_trials='151-200',known_noise=.1,
              resolution=args.resolution,limit=args.limit,
              near_optimal='within 1 total prefix NLL, descriptive not confidence region; max 8 representative actual grid points',
              fitted_model='unknown alpha, reward weight, sign; known family/noise/Q scale; fixed after trial 150',
              common_history='all 7 source histories equally weighted within base participant; target weight and oracle parameters used only for replay',
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    sp=args.output/f'{args.phase}_protocol.json'
    if sp.exists():
        old=json.loads(sp.read_text())
        if old!=spec: raise RuntimeError('Refusing to resume with changed protocol')
    else: sp.write_text(json.dumps(spec,indent=2))
    if args.phase=='common': common_history(args.output,scale)
    else:
        report={}
        for w in WEIGHTS:
            name=f'reward_w{round(w*100):03d}'
            d=load(w)
            maps=np.load(ROOT/name/'validation_assay.npz')['donor_mappings']
            if args.limit:
                from run_weight_curve_a import derangements
                d={k:v[:args.limit] for k,v in d.items()}
                maps=derangements(args.limit,20,20260916)
            fp=args.output/f'{name}_fits.npz'
            if fp.exists(): fits=dict(np.load(fp))
            else:
                fits=grid_fit(d['action'][:,:150],d['reward'][:,:150],scale,args.resolution)
                np.savez_compressed(fp,**fits)
            result,arrays=fit_assay(d,fits,scale,maps)
            report[name]=result
            np.savez_compressed(args.output/f'{name}_responses.npz',**arrays)
            (args.output/'fitted_summary.json').write_text(json.dumps(report,indent=2))
            print('FITTED_ASSAY',name,flush=True)
    (args.output/f'{args.phase}_COMPLETE.json').write_text(json.dumps(dict(complete=True,test_used=False)))


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
