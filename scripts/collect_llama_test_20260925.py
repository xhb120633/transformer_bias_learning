"""Strict independent-test main-assay collection; never mixes validation exports."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

WEIGHTS = ('000','010','030','050','070','090','100')
MODELS = ('oracle','fit','rw','gru','transformer')
METRICS = ('intact_nll','donor_nll','delta_nll','response_error')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def prob(lp): return np.exp(lp.astype(float)-logsumexp(lp.astype(float),axis=-1,keepdims=True))
def nll(p,a): return -np.log(np.take_along_axis(p,a[...,None],-1)[...,0].clip(1e-300))
def summarize(x,seed=731):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    b=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())
def metrics(p,q,op,oq,a):
    base=nll(p,a).mean(1);donor=nll(q,a[None]).mean((0,2))
    return dict(intact_nll=base,donor_nll=donor,delta_nll=donor-base,
                response_error=(.5*np.abs((q-p[None])-(oq-op[None])).sum(-1)).mean((0,2)))

def check_row(row,i,ids,actions,maps,task,mode):
    assert int(row['index'])==i and str(row['participant'])==str(ids[i])
    np.testing.assert_array_equal(row['actions'],actions[i])
    np.testing.assert_array_equal(row['donor_indices'],maps[:,i])
    shape=(200,4) if task=='restless' else (8,20,25)
    assert row['intact_logp'].shape==shape
    if mode=='full': assert row['donor_logp'].shape==((20,50,4) if task=='restless' else (20,2,20,25))
    for key,value in row.items():
        if key.endswith('logp'):
            assert np.isfinite(value).all() and np.all(np.exp(value).sum(-1)<=1.0001)
    if task=='spatial' and mode=='full':
        np.testing.assert_allclose(row['donor_logp'][:,:,0],np.broadcast_to(row['intact_logp'][None,6:,0],(20,2,25)),atol=.003,rtol=0)

def collect(task,args,manifest,frozen):
    part='task_a' if task=='restless' else 'task_b';n=250 if task=='restless' else 50;arms=4 if task=='restless' else 25
    out=args.output/task;out.mkdir(parents=True,exist_ok=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(split='test',main_complete=False,scientific_audit_passed=False,status='collection_in_progress')))
    non=args.nonllm
    audit=read(non/'AUDIT.json');assert audit['split']=='test' and audit['complete']
    assert audit['tasks'][task]['neural_caches']==84
    A=copy.deepcopy(read(non/f'{task}_A_summary.json'));B=copy.deepcopy(read(non/f'{task}_B_summary.json'))
    Aall=copy.deepcopy(read(non/f'{task}_A_alltrials_summary.json'))
    expected_keys={str(int(w)/100) for w in WEIGHTS}
    assert set(A)==set(B)==expected_keys
    provenance={'split':'test','input_hashes':{},'weights':{},'frozen_inputs':frozen,
                'collector_sha256':sha(Path(__file__)),'payload_manifest_sha256':sha(args.payload/'MANIFEST.json'),
                'frozen_inputs_sha256':sha(args.input/'FROZEN_INPUTS.json'),
                'nonllm_audit_sha256':sha(non/'AUDIT.json')}
    script=Path('scripts')/('eval_weight_llama.py' if task=='restless' else 'eval_spatial_llama.py')
    frozen_script='scripts/'+('eval_weight_llama_test_20260925.py' if task=='restless' else 'eval_spatial_llama_test_20260925.py')
    assert sha(script)==frozen['scripts'][frozen_script]
    for w in WEIGHTS:
        name='reward_w'+w;key=str(int(w)/100)
        dmfile=non/f'{task}_donor_mappings.npz'
        with np.load(dmfile) as z: maps=z['donor_mappings'];ids=z['participants'].astype(str)
        assert maps.shape==(20,n) and len(ids)==n
        for m in maps:
            np.testing.assert_array_equal(np.sort(m),np.arange(n));assert np.all(m!=np.arange(n))
        if task=='restless':
            raw=Path('outputs/weight_curve_a_20260915')/name/'test_000.npz'
            with np.load(raw) as z:actions=z['action'];np.testing.assert_array_equal(ids,z['base_participant_id'].astype(str))
            payload_maps=args.payload/part/(name+'_maps.npz')
            with np.load(payload_maps) as z:
                np.testing.assert_array_equal(z['participants'].astype(str),ids);np.testing.assert_array_equal(z['donor_mappings'],maps)
        else:
            raw=Path('outputs/spatial_pilot_v1')/name/'test_audit.npz'
            with np.load(raw) as z:actions=z['actions'];np.testing.assert_array_equal(ids,z['participant'].astype(str))
            payload=read(args.payload/part/(name+'.json'))
            assert payload['split']=='test' and payload['test_used']
            np.testing.assert_array_equal(payload['actions'],actions);np.testing.assert_array_equal(payload['donor_maps'],maps)
            np.testing.assert_array_equal([str(r['participant']) for r in payload['records']],ids)
        assert audit['source_sha256'][str(raw)]==sha(raw)
        exports={};manifests={}
        for mode in ('full','choice' if task=='restless' else 'choice_only'):
            folder=args.input/part/'production'/name/mode
            adapter=('outputs/'+('weight_llama_sft_20260916' if mode=='full' else 'weight_choice_only_sft_20260916')+'/'+name+'_seed100' if task=='restless' else 'outputs/spatial_llama_v1/'+name+'_'+mode+'_seed100')
            adapter_hashes=frozen['adapters'][adapter]
            assert all(len(adapter_hashes[k])==64 for k in ('adapter_model.safetensors','adapter_config.json'))
            source=args.payload/part/mode/(name+'.jsonl') if task=='restless' else args.payload/part/(name+'.json')
            completion=sorted(folder.glob('COMPLETE_shard*.json')) if task=='restless' else [folder/'COMPLETE.json']
            assert len(completion)==(5 if task=='restless' and mode=='full' else 1), (folder,'incomplete manifests')
            covered=[];meta=[]
            for cp in completion:
                c=read(cp);assert c['split']=='test' and not c['smoke'] and c['mode']==mode and c['weight']==w
                assert c['source_sha256']==sha(source) and c['script_sha256']==sha(script)
                assert c['script_sha256']==frozen['scripts'][frozen_script]
                if task=='restless':
                    assert c['adapter'].replace('\\','/').endswith('/'+adapter) or c['adapter'].replace('\\','/')==adapter
                    assert c['maps_sha256']==sha(payload_maps)
                    assert c['shards']==(5 if mode=='full' else 1)
                    np.testing.assert_array_equal(c['indices'],np.arange(n)[c['shard']::c['shards']])
                    causal_error=c['causal_logp_max_error']
                    if causal_error is None:
                        # Resumes skip the already published first participant; the
                        # original job's successful probe remains in its raw log.
                        original={('000',4):4,('050',2):17,('090',3):28}
                        assert mode=='full' and (w,c['shard']) in original
                        job=original[(w,c['shard'])]
                        log=args.input/'resume_audit_logs'/f'test-A-full_13587155_{job}.out'
                        contents=log.read_text(encoding='utf-8')
                        assert f'family=task_a mode=full weight={w} shard={c["shard"]}/5' in contents
                        errors=re.findall(r'^SAME_SHAPE_CAUSAL_AUDIT ([0-9.eE+-]+)$',contents,re.M)
                        assert len(errors)==1 and f'PARTICIPANT_COMPLETE {w} full {c["shard"]} ' in contents
                        causal_error=float(errors[0])
                        provenance['input_hashes'][str(log)]=sha(log)
                        c['resume_causal_evidence']={'log':str(log),'sha256':sha(log),'error':causal_error}
                    assert np.isfinite(causal_error) and 0<=causal_error<.003
                    if mode=='full':assert c['operations']==['intact','donor']
                    covered+=c['indices']
                else:
                    assert c['complete'] and c['test_used'] and c['fixed_length']==1024 and c['causal_max_difference']<.003
                    assert c['adapter_config_sha256']==adapter_hashes['adapter_config.json']
                provenance['input_hashes'][str(cp)]=sha(cp);meta.append(c)
            if task=='restless':assert sorted(covered)==list(range(n))
            records=[]
            for i in range(n):
                path=folder/f'participant_{i:03d}.npz'
                with np.load(path,allow_pickle=False) as z:r=dict(z)
                check_row(r,i,ids,actions,maps,task,mode);records.append(r)
                provenance['input_hashes'][str(path)]=sha(path)
            exports[mode]={k:np.stack([r[k] for r in records],axis=1 if k=='donor_logp' else 0) for k in ('intact_logp','donor_logp') if k in records[0]}
            manifests[mode]=meta
        f=exports['full'];c=exports['choice' if task=='restless' else 'choice_only']
        allp=prob(f['intact_logp']);allc=prob(c['intact_logp']);q=prob(f['donor_logp']).reshape(20,n,-1,arms)
        select=lambda x: x[:,150:] if task=='restless' else x[:,6:].reshape(n,40,arms)
        p=select(allp);cp=select(allc);a=actions[:,150:] if task=='restless' else actions[:,6:].reshape(n,40)
        refpath=non/f'{task}_{name}_references.npz';vecpath=non/f'{task}_{name}_vectors.npz'
        with np.load(refpath) as z:ref=dict(z)
        if task=='restless': assert int(ref['grid_resolution'])==81
        with np.load(vecpath) as z:base=dict(z)
        op=ref['oracle_intact'].reshape(n,-1,arms);oq=ref['oracle_donor'].reshape(20,n,-1,arms)
        for model in ('oracle','fit','rw'):
            for k,v in metrics(ref[model+'_intact'].reshape(n,-1,arms),ref[model+'_donor'].reshape(20,n,-1,arms),op,oq,a).items():
                np.testing.assert_allclose(base[model+'_'+k],v,atol=1e-10)
        for model in MODELS:
            for metric in METRICS:
                assert model+'_'+metric in B[key]
                np.testing.assert_allclose(B[key][model+'_'+metric]['mean'],base[model+'_'+metric].mean(),atol=1e-10)
        for family in ('gru','transformer'):
            for mode in ('full','choice'):
                assert family+'_'+mode+'_nll' in A[key]['all']
        v=metrics(p,q,op,oq,a);v['full_nll']=v['intact_nll'];v['choice_nll']=nll(cp,a).mean(1)
        v['reward_predictive_gain']=v['choice_nll']-v['intact_nll']
        for mode,lp in [('full',f['intact_logp']),('choice',c['intact_logp'])]:
            sl=select(lp)
            v[mode+'_token_nll']=-np.take_along_axis(sl,a[...,None],-1)[...,0].mean(1)
            v[mode+'_valid_mass']=np.exp(sl).sum(-1).mean(1)
        for model in MODELS:
            for k in ('intact_nll','response_error','delta_nll'):v[k+'_minus_'+model]=v[k]-base[model+'_'+k]
        for k,x in v.items():B[key]['llama_'+k]=summarize(x)
        for k in ('full_nll','choice_nll','reward_predictive_gain'):
            A[key]['all']['llama_'+k]=summarize(v[k])
        # Do not carry stale subgroup summaries when only all-participant curves are requested.
        A[key]={'all':A[key]['all']}
        for mode,probs in [('full',allp),('choice',allc)]:
            x=nll(probs.reshape(n,-1,arms),actions.reshape(n,-1)).mean(1)
            Aall[key]['all']['llama_'+mode+'_nll']=summarize(x)
        np.testing.assert_allclose(v['donor_nll']-v['intact_nll'],v['delta_nll'],atol=1e-12)
        np.savez_compressed(out/(name+'_llama.npz'),participants=ids,actions=actions,donor_maps=maps,intact=p,choice=cp,donor=q,**v)
        provenance['weights'][key]={'participants':n,'donors':20,'manifests':manifests,'identity_actions_maps_verified':True}
        for pp in (refpath,vecpath,dmfile,raw):provenance['input_hashes'][str(pp)]=sha(pp)
    for file,data in [('A_summary.json',A),('B_summary.json',B),('A_alltrials_summary.json',Aall),('audit.json',provenance)]:
        (out/file).write_text(json.dumps(data,indent=2),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(split='test',test_used=True,scientific_audit_passed=True,main_complete=True,models=[*MODELS,'llama'],weights=list(WEIGHTS),donors=20,participants_per_weight=n,scoring='trials151-200' if task=='restless' else 'maps7-8',appendix_operations_included=False),indent=2))
    print('AUDITED_TEST_COMPLETE',task,out,flush=True)

def self_test():
    rng=np.random.default_rng(41);lp=np.log(rng.dirichlet(np.ones(4),size=(3,50)))
    p=prob(lp);q=np.repeat(p[None],20,axis=0);a=np.zeros((3,50),int)
    v=metrics(p,q,p,q,a)
    np.testing.assert_allclose(v['delta_nll'],0,atol=1e-14);np.testing.assert_array_equal(v['response_error'],0)
    row=dict(index=0,participant='id',actions=np.zeros(200,int),donor_indices=np.zeros(20,int),intact_logp=np.full((200,4),-np.log(4)),donor_logp=np.full((20,50,4),-np.log(4)))
    check_row(row,0,['id'],np.zeros((1,200),int),np.zeros((20,1),int),'restless','full')
    row['donor_logp']=row['donor_logp'][:,:49]
    try:check_row(row,0,['id'],np.zeros((1,200),int),np.zeros((20,1),int),'restless','full')
    except AssertionError:pass
    else:raise AssertionError('Malformed donor shape was accepted')
    print('SELF_TEST_PASS')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['restless','spatial','both'],default='both')
    ap.add_argument('--input',type=Path,default=Path('outputs/llama_test_20260925'))
    ap.add_argument('--payload',type=Path,default=Path('data/llama_test_20260925'))
    ap.add_argument('--nonllm',type=Path,default=Path('outputs/test_nonllm_20260925'))
    ap.add_argument('--output',type=Path,default=Path('outputs/test_main_integrated_20260925'))
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return
    manifest=read(args.payload/'MANIFEST.json');assert manifest['split']=='test' and manifest['no_test_model_selection']
    for p,h in manifest['files'].items():assert sha(args.payload/Path(p))==h,(p,'payload changed')
    frozen=read(args.input/'FROZEN_INPUTS.json')
    assert frozen['split']=='test' and len(frozen['adapters'])==28
    assert frozen['payload_manifest_sha256']==sha(args.payload/'MANIFEST.json')
    for task in (('restless','spatial') if args.task=='both' else (args.task,)):collect(task,args,manifest,frozen)

if __name__=='__main__':main()
