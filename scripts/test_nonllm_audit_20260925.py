"""Audit independent-test exports without any model selection."""
import json
import numpy as np
from test_nonllm_20260925 import OUT,RA,RB,WEIGHTS,SEEDS,name,sha,read,maps,rows
from run_weight_curve_a import replay
from run_reference_followup import params
from ablate_spatial_references import nll

def main():
    report={'split':'test','tasks':{},'optimizer_failures':[],'source_sha256':{}}
    for task,root in [('restless',RA),('spatial',RB)]:
        ncache=0;dm=maps(task);counts=[]
        for w in WEIGHTS:
            folder=root/name(w);d=read(w,task);idkey='base_participant_id' if task=='restless' else 'participant'
            ids=set(d[idkey]);counts.append(len(ids))
            for split in ('train','val'):
                path=folder/(f'{split}_000.npz' if task=='restless' else f'{split}_audit.npz')
                with np.load(path) as z:assert not ids.intersection(z[idkey].tolist())
            source=folder/('test_000.npz' if task=='restless' else 'test_audit.npz');report['source_sha256'][str(source)]=sha(source)
            assert np.all(dm!=np.arange(len(ids)))
            for m in dm:np.testing.assert_array_equal(np.sort(m),np.arange(len(ids)))
            ref=np.load(OUT/f'{task}_{name(w)}_references.npz')
            if w==0:np.testing.assert_array_equal(ref['oracle_donor'],np.broadcast_to(ref['oracle_intact'],ref['oracle_donor'].shape))
            if task=='restless':
                assert int(ref['grid_resolution'])==81
                fits=np.load(OUT/f'restless_{name(w)}_fits_fine.npz');pa=fits['parameters'][:,0]
                fitd=params(d,pa[:,0],pa[:,1],pa[:,2]);scale=json.loads((RA/'protocol.json').read_text())['q_scale']
                loss=nll(replay(fitd,d['reward'],scale),d['action'])[:,:150].sum(1)
                # Stored float32 lapse differs from fitting's exact .1 by ~1e-9;
                # this accumulates <1e-5 nats across the entire 150-trial prefix.
                np.testing.assert_allclose(loss,fits['fit_total_nll'][:,0],atol=1e-5,rtol=1e-7)
            else:
                assert [r['participant'] for r in rows(w)]==d['participant'].tolist()
                for i in range(len(ids)):
                    f=json.loads((OUT/f'spatial_fit_{name(w)}_{i:03d}.json').read_text())
                    assert f['participant']==d['participant'][i]
                    if not f['models']['mixture']['optimizer_success']:report['optimizer_failures'].append([w,i])
            for family in ('gru','transformer'):
                for mode in ('full','choice_only'):
                    for seed in SEEDS:
                        path=OUT/f'{task}_{name(w)}_{family}_{mode}_seed{seed}.npz'
                        if not path.exists():continue
                        with np.load(path) as z:
                            for k in ('intact','donor'):
                                if k not in z:continue
                                p=z[k];assert np.isfinite(p).all() and (p>0).all()
                                np.testing.assert_allclose(p.sum(-1),1,atol=2e-6)
                            assert len(str(z['checkpoint_sha256']))==64
                            if task=='spatial' and mode=='full':np.testing.assert_allclose(z['donor'][:,:,:,0],np.broadcast_to(z['intact'][None,:,6:,0],z['donor'][:,:,:,0].shape),atol=2e-5,rtol=2e-5)
                        ncache+=1
        report['tasks'][task]={'neural_caches':ncache,'expected_caches':84,'participants_per_weight':counts,'donor_sha256':sha(OUT/f'{task}_donor_mappings.npz')}
    report['complete']=all(v['neural_caches']==84 for v in report['tasks'].values()) and not report['optimizer_failures']
    (OUT/'AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
