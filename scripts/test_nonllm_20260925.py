"""Frozen-checkpoint independent test evaluation. Never selects models on test."""
import argparse, copy, hashlib, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from run_weight_curve_a import WEIGHTS, derangements, replay
from run_reference_followup import grid_fit, params, groups
from ablate_spatial_references import nll, summarize
from run_rw_baseline import rw, metrics
from analyze_spatial_neural import predict as spatial_predict, load_model as spatial_model, fit_one
from mechcal.models.gru import CausalGRU, GRUConfig
from mechcal.models import CausalTransformer, TransformerConfig
from mechcal.generators.spatial_mixture import SpatialConfig, replay as spatial_replay
from mechcal.analysis.reward_ablation import choice_only_ablation
from analyze_trainfit_baselines import predict as baseline_predict

OUT=Path('outputs/test_nonllm_20260925')
RA=Path('outputs/weight_curve_a_20260915'); RB=Path('outputs/spatial_pilot_v1')
SEEDS=(11,22,33)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def name(w):return f'reward_w{round(w*100):03d}'
def read(w,task):
    root=RA if task=='restless' else RB
    return dict(np.load(root/name(w)/('test_000.npz' if task=='restless' else 'test_audit.npz')))
def rows(w):return [json.loads(s) for s in (RB/name(w)/'test_observable.jsonl').read_text().splitlines()]
def maps(task):return np.load(OUT/f'{task}_donor_mappings.npz')['donor_mappings']
def prepare():
    OUT.mkdir(exist_ok=True)
    for task,seed in [('restless',20260917),('spatial',81073)]:
        d=read(0,task);ids=d['base_participant_id' if task=='restless' else 'participant']
        mappings=derangements(len(ids),20,seed)
        path=OUT/f'{task}_donor_mappings.npz'
        if path.exists():np.testing.assert_array_equal(np.load(path)['donor_mappings'],mappings)
        else:np.savez_compressed(path,donor_mappings=mappings,participants=ids,seed=seed)
        manifest={'split':'test','seed':seed,'participants':ids.tolist(),'donor_mappings':mappings.tolist(),'npz_sha256':sha(path)}
        (OUT/f'{task}_donor_mappings.json').write_text(json.dumps(manifest,indent=2))
        for w in WEIGHTS:
            q=read(w,task);np.testing.assert_array_equal(ids,q['base_participant_id' if task=='restless' else 'participant'])
        print('MAPPINGS',task,path,sha(path),flush=True)
    (OUT/'protocol.json').write_text(json.dumps(dict(split='test',weights=WEIGHTS,seeds=SEEDS,checkpoint_selection='Existing validation-selected best.pt; frozen before test',A='Suffix151-200 / maps7-8; all-trial metrics additionally saved',B='Same suffix; cumulative donor chosen rewards; actual choices and spatial cue retained',individual_fit='Restless first150 trials grid81 (82 alphas,101 weights,2 signs), matching manuscript; spatial first6 maps same three-start fit',donors=20),indent=2))

@torch.inference_mode()
def rest_predict(model,tokens,mode):
    p=[]
    for b in tokens.split(32):
        z=model(b[:,:-1].cuda()).float()
        z=z[:,0::2,1:5] if mode=='full' else z[:,:,1:5]
        p.append(z.softmax(-1).cpu().numpy())
    return np.concatenate(p)

def neural(task):
    for w in WEIGHTS:
        d=read(w,task);dm=maps(task)
        rr=rows(w) if task=='spatial' else None
        for family in ('gru','transformer'):
            for mode in ('full','choice_only'):
                for seed in SEEDS:
                    dst=OUT/f'{task}_{name(w)}_{family}_{mode}_seed{seed}.npz'
                    if dst.exists():continue
                    if task=='restless':
                        root=Path('outputs')/('weight_neural_full_20260916' if mode=='full' else 'weight_neural_choice_only_20260916')
                        ckpath=root/f'{family}_{name(w)}_{mode}_seed{seed}'/'best.pt'
                        ck=torch.load(ckpath,map_location='cpu',weights_only=True)
                        model=(CausalGRU(GRUConfig(**ck['model_config'])) if family=='gru' else CausalTransformer(TransformerConfig(**ck['model_config']))).cuda().eval()
                        model.load_state_dict(ck['model_state']);digest=sha(ckpath)
                        tok=torch.tensor(d['tokens'].astype(np.int64))
                        if mode=='choice_only':tok=choice_only_ablation(tok)[0]
                        p=rest_predict(model,tok,mode);q=[]
                        if mode=='full':
                            for m in dm:
                                changed=tok.clone();changed[:,2::2]=tok[m,2::2]
                                altered=rest_predict(model,changed,mode)
                                np.testing.assert_allclose(altered[:,0],p[:,0],atol=2e-5)
                                q.append(altered[:,150:])
                    else:
                        model,capacity,digest=spatial_model(family,mode,name(w),seed)
                        p=spatial_predict(model,[ep for r in rr for ep in r['rounds']],mode,capacity).reshape(len(rr),8,20,25);q=[]
                        if mode=='full':
                            for m in dm:
                                altered=[]
                                for i in range(len(rr)):
                                    for b in (6,7):
                                        ep=copy.deepcopy(rr[i]['rounds'][b]);ep['rewards']=rr[int(m[i])]['rounds'][b]['rewards'];altered.append(ep)
                                v=spatial_predict(model,altered,mode,capacity).reshape(len(rr),2,20,25)
                                np.testing.assert_allclose(v[:,:,0],p[:,6:,0],atol=2e-5,rtol=2e-5);q.append(v)
                    export=dict(intact=p,checkpoint_sha256=digest)
                    if q:export['donor']=np.array(q)
                    np.savez_compressed(dst,**export)
                    del model;torch.cuda.empty_cache();print('NEURAL',dst.name,flush=True)

def spatial_fits():
    config=json.loads((RB/'protocol.json').read_text())['config'];tasks=[]
    for w in WEIGHTS:
        for i,r in enumerate(rows(w)):
            dst=OUT/f'spatial_fit_{name(w)}_{i:03d}.json'
            if not dst.exists():tasks.append((dst,(w,r,config,31415+i)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (dst,_),r in zip(tasks,pool.map(fit_one,[x[1] for x in tasks])):
            dst.write_text(json.dumps(r,indent=2));print('FIT',dst.name,r['models']['mixture']['optimizer_success'],flush=True)

def references(task):
    for w in WEIGHTS:
        dst=OUT/f'{task}_{name(w)}_references.npz'
        if dst.exists():
            if task!='restless':continue
            with np.load(dst) as old:
                if 'grid_resolution' in old and int(old['grid_resolution'])==81:continue
        d=read(w,task);dm=maps(task);save={}
        if task=='restless':
            a=d['action'];r=d['reward'];scale=json.loads((RA/'protocol.json').read_text())['q_scale']
            fp=OUT/f'restless_{name(w)}_fits_fine.npz'
            if not fp.exists():np.savez_compressed(fp,**grid_fit(a[:,:150],r[:,:150],scale,81))
            save['grid_resolution']=81
            pars=np.load(fp)['parameters'][:,0];fitted=params(d,pars[:,0],pars[:,1],pars[:,2])
            for ref,rd in [('oracle',d),('fit',fitted)]:
                p=replay(rd,r,scale);q=np.array([replay(rd,r[m],scale)[:,150:] for m in dm]);save[ref+'_all']=p;save[ref+'_intact']=p[:,150:];save[ref+'_donor']=q
            np.testing.assert_allclose(save['oracle_all'],d['choice_probability'],atol=1e-6)
            theta=json.loads((Path('outputs/rw_pooled_baseline_v1')/f'restless_{name(w)}_fit.json').read_text())['theta']
            p=rw(theta,a,r,4);save['rw_all']=p;save['rw_intact']=p[:,150:];save['rw_donor']=np.array([rw(theta,a,r[m],4)[:,150:] for m in dm])
            bf=json.loads(Path('outputs/task1_diagnostics_20260916/baseline_fits.json').read_text())[str(w)]
            for k,p in baseline_predict(a,bf).items():save['baseline_'+k]=p[:,149:]
        else:
            c=SpatialConfig(**json.loads((RB/'protocol.json').read_text())['config']);a=d['actions'];r=d['rewards'];n=len(a)
            for ref in ('oracle','fit'):
                p=np.empty((n,2,20,25));q=np.empty((20,n,2,20,25))
                for i in range(n):
                    if ref=='oracle':pars={k:float(d[k][i]) for k in ('gp_length','reward_temperature','local_temperature')};weight=w
                    else:
                        fit=json.loads((OUT/f'spatial_fit_{name(w)}_{i:03d}.json').read_text())['models']['mixture']
                        pars=fit['parameters'].copy();weight=pars.pop('weight')
                    for j,b in enumerate((6,7)):
                        args=(c,pars,weight,int(d['cue_arm'][i,b]),float(d['cue_reward'][i,b]),a[i,b])
                        p[i,j]=spatial_replay(*args,r[i,b])
                        for k,m in enumerate(dm):q[k,i,j]=spatial_replay(*args,r[m[i],b])
                save[ref+'_intact']=p;save[ref+'_donor']=q
            save['oracle_all']=d['probabilities'];np.testing.assert_allclose(save['oracle_intact'],d['probabilities'][:,6:],atol=1e-12)
            theta=json.loads((Path('outputs/rw_pooled_baseline_v1')/f'spatial_{name(w)}_fit.json').read_text())['theta'];cue=(d['cue_arm'].reshape(-1),d['cue_reward'].reshape(-1))
            p=rw(theta,a.reshape(-1,20),r.reshape(-1,20),25,cue).reshape(n,8,20,25);save['rw_all']=p;save['rw_intact']=p[:,6:]
            save['rw_donor']=np.array([rw(theta,a.reshape(-1,20),r[m].reshape(-1,20),25,cue).reshape(n,8,20,25)[:,6:] for m in dm])
        if w==0:np.testing.assert_array_equal(save['oracle_donor'],np.broadcast_to(save['oracle_intact'],save['oracle_donor'].shape))
        np.savez_compressed(dst,**save);print('REFERENCES',task,w,flush=True)

def aggregate():
    for task in ('restless','spatial'):
        result={};bresult={};allresult={}
        for w in WEIGHTS:
            rp=OUT/f'{task}_{name(w)}_references.npz'
            if not rp.exists():continue
            d=read(w,task);a=d['action'][:,150:] if task=='restless' else d['actions'][:,6:].reshape(-1,40);n=len(a);arms=4 if task=='restless' else 25
            ref=dict(np.load(rp));op=ref['oracle_intact'].reshape(n,-1,arms);oq=ref['oracle_donor'].reshape(20,n,-1,arms);vectors={};allvectors={}
            alla=d['action'] if task=='restless' else d['actions'].reshape(n,-1)
            for model in ('oracle','rw'):
                allvectors[model+'_nll']=nll(ref[model+'_all'].reshape(n,-1,arms),alla).mean(1)
            for model in ('oracle','fit','rw'):
                m=metrics(ref[model+'_intact'].reshape(n,-1,arms),ref[model+'_donor'].reshape(20,n,-1,arms),op,oq,a)
                vectors.update({model+'_'+k:v for k,v in m.items()})
            for k,p in ref.items():
                if k.startswith('baseline_'):vectors[k+'_nll']=nll(p,a).mean(1)
            for family in ('gru','transformer'):
                for mode in ('full','choice_only'):
                    paths=[OUT/f'{task}_{name(w)}_{family}_{mode}_seed{seed}.npz' for seed in SEEDS]
                    if not all(p.exists() for p in paths):continue
                    ms=[]
                    for path in paths:
                        z=np.load(path);p=(z['intact'][:,150:] if task=='restless' else z['intact'][:,6:].reshape(n,40,arms)).astype(float);p/=p.sum(-1,keepdims=True)
                        if mode=='full':
                            q=z['donor'].reshape(20,n,-1,arms).astype(float);q/=q.sum(-1,keepdims=True)
                            ms.append(metrics(p,q,op,oq,a))
                        else:ms.append({'nll':nll(p,a).mean(1)})
                    prefix=family if mode=='full' else family+'_choice'
                    vectors.update({prefix+'_'+k:np.mean([m[k] for m in ms],0) for k in ms[0]})
                    vectors[family+('_full_nll' if mode=='full' else '_choice_nll')]=np.mean([m['intact_nll' if mode=='full' else 'nll'] for m in ms],0)
                    allvectors[family+('_full_nll' if mode=='full' else '_choice_nll')]=np.mean([nll(np.load(p)['intact'].reshape(n,-1,arms),alla).mean(1) for p in paths],0)
            np.savez_compressed(OUT/f'{task}_{name(w)}_vectors.npz',**vectors)
            vectors['oracle_nll']=vectors['oracle_intact_nll']
            result[str(w)]=groups(vectors,d['kernel_sign']) if task=='restless' else {'all':{k:summarize(v) for k,v in vectors.items()}}
            bresult[str(w)]={k:summarize(v) for k,v in vectors.items() if not k.startswith('baseline_')}
            allresult[str(w)]={'all':{k:summarize(v) for k,v in allvectors.items()}}
        (OUT/f'{task}_summary.json').write_text(json.dumps(result,indent=2))
        (OUT/f'{task}_A_summary.json').write_text(json.dumps(result,indent=2))
        (OUT/f'{task}_B_summary.json').write_text(json.dumps(bresult,indent=2))
        (OUT/f'{task}_A_alltrials_summary.json').write_text(json.dumps(allresult,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','neural','spatial_fits','references','aggregate']);ap.add_argument('--task',choices=['restless','spatial']);args=ap.parse_args();torch.set_num_threads(4)
    if args.phase=='prepare':prepare()
    elif args.phase=='neural':neural(args.task)
    elif args.phase=='spatial_fits':spatial_fits()
    elif args.phase=='references':references(args.task)
    else:aggregate()
