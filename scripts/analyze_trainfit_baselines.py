"""Corrected section A controls: train-only fitting and lag selection."""
import json
from pathlib import Path
import numpy as np
from analyze_frequency_baselines import DATA, score
from finish_calibration_and_local import interval
from mechcal.training.restless_dataset import RestlessTranscriptDataset

OUT = Path('outputs/trainfit_baselines_20260915')


def lag_predict(a, lag, repeat):
    n,t = a.shape
    p = np.full((n,t-1,4), .25)
    p[:,lag-1:] = (1-repeat)/3
    np.put_along_axis(p[:,lag-1:], a[:,:t-lag,None], repeat, -1)
    return p


def fit(a):
    freq = np.bincount(a[:,1:].ravel(),minlength=4).astype(float)
    freq /= freq.sum()
    transition = np.ones((4,4))
    np.add.at(transition,(a[:,:-1].ravel(),a[:,1:].ravel()),1)
    transition /= transition.sum(-1,keepdims=True)
    candidates = []
    for lag in range(1,21):
        repeat = float((a[:,lag:]==a[:,:-lag]).mean())
        loss = float(score(lag_predict(a,lag,repeat),a[:,1:])['nll'].mean())
        candidates.append({'lag':lag,'repeat':repeat,'train_nll':loss})
    selected = min(candidates,key=lambda x:(x['train_nll'],x['lag']))
    return {'arm_frequency':freq.tolist(),'transition':transition.tolist(),
            'stay_probability':candidates[0]['repeat'],'selected':selected,'candidates':candidates}


def predict(a, fitted):
    n,t = a.shape; chosen=fitted['selected']
    return {'uniform':np.full((n,t-1,4),.25),
            'static_arm_frequency':np.broadcast_to(fitted['arm_frequency'],(n,t-1,4)).copy(),
            'static_stay_probability':lag_predict(a,1,fitted['stay_probability']),
            'static_transition':np.asarray(fitted['transition'])[a[:,:-1]],
            'train_selected_lag':lag_predict(a,chosen['lag'],chosen['repeat'])}


def main():
    OUT.mkdir(exist_ok=True)
    result={'protocol':'All static parameters and lag selected on training only. Lag candidates 1..20; criterion training NLL on common trials 2..200, uniform when lag unavailable, smallest lag tie break. Validation frozen evaluation only. Test frozen evaluation. Laplace unit transition pseudocount. Baselines exploratory, not pristine confirmation after earlier test inspection. CIs over participants conditional on fitted predictors. Online baselines remain separate in earlier output, not merged into static table.', 'conditions':{}}
    for c in ('reward_dominant','balanced','choice_dominant'):
        ds={s:RestlessTranscriptDataset(DATA,s,c) for s in ('train','val','test')}
        ids={s:set(d.audit.base_participant_id) for s,d in ds.items()}
        assert not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test']
        actions={s:d.tokens[:,1::2].numpy()-1 for s,d in ds.items()}
        fitted=fit(actions['train'])
        # Prefix causality audit: later actions must not affect earlier predictions.
        a=actions['test']; altered=a.copy(); altered[:,100:]=(altered[:,100:]+1)%4
        for k,p in predict(a,fitted).items():
            assert np.array_equal(p[:,:100],predict(altered,fitted)[k][:,:100])
        validation={k:{m:float(v.mean()) for m,v in score(p,actions['val'][:,1:]).items()} for k,p in predict(actions['val'],fitted).items()}
        metrics={k:score(p,a[:,1:]) for k,p in predict(a,fitted).items()}
        for family,root in [('gru_choice_only','gru_suite_20260907'),('transformer_choice_only','transformer_choice_only_20260915')]:
            seeds=[]
            for seed in (11,22,33):
                name=f'{c}_choice_only_seed{seed}' if family.startswith('gru') else f'{c}_seed{seed}'
                with np.load(Path('outputs')/root/name/'predictions.npz') as z:
                    assert np.array_equal(z['participants'],ds['test'].audit.base_participant_id) and np.array_equal(z['actions'],a)
                    p=z['intact_choice_probabilities'].astype(float)
                p/=p.sum(-1,keepdims=True)
                seeds.append(score(p[:,1:],a[:,1:]))
            metrics[family]={k:np.mean([s[k] for s in seeds],axis=0) for k in seeds[0]}
        strata={}
        for name,mask in [('all',np.ones(len(a),bool)),('positive',ds['test'].audit.kernel_sign==1),('negative',ds['test'].audit.kernel_sign==-1)]:
            strata[name]={model:{k:interval(v[mask]) for k,v in ms.items()} for model,ms in metrics.items()}
            for family in ('gru_choice_only','transformer_choice_only'):
                strata[name][family+'_minus_baselines']={b:{k:interval((metrics[family][k]-metrics[b][k])[mask]) for k in ('nll','argmax_accuracy')} for b in predict(a,fitted)}
        result['conditions'][c]={'fit':fitted,'validation':validation,'strata':strata}
        np.savez_compressed(OUT/f'{c}_vectors.npz',participants=ds['test'].audit.base_participant_id,**{f'{model}_{k}':v for model,ms in metrics.items() for k,v in ms.items()})
        print(c,'selected',fitted['selected'],json.dumps({model:{k:float(v.mean()) for k,v in ms.items()} for model,ms in metrics.items()}),flush=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
