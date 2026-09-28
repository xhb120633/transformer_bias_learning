"""Static and causal online frequency controls for choice-only prediction."""
import json
from pathlib import Path
import numpy as np
from mechcal.training.restless_dataset import RestlessTranscriptDataset
from finish_calibration_and_local import nll, interval

DATA = Path('outputs/restless_pooled_eps10_1000_20260827')
OUT = Path('outputs/frequency_baselines_20260915')


def score(p, a):
    assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(-1), 1)
    top = p == p.max(-1, keepdims=True)
    # Uniform tie breaking avoids an arbitrary preferred arm.
    hit = np.take_along_axis(top, a[..., None], -1)[..., 0]/top.sum(-1)
    return {'nll': nll(p, a).mean(1), 'argmax_accuracy': hit.mean(1),
            'sampled_expected_accuracy': np.take_along_axis(p, a[..., None], -1)[..., 0].mean(1)}


def main():
    OUT.mkdir(exist_ok=True)
    result = {'protocol': 'Test trials 2-200; static parameters estimated on validation only. Online predictors use strictly preceding choices; Dirichlet(1,1,1,1) arm prior and Beta(1,1) stay prior fixed without test tuning. Seed-mean GRU scores, not ensemble. CIs bootstrap participants, condition on fitted models. Argmax accuracy uses uniform tie breaking; sampled accuracy is expected accuracy when drawing from model probabilities.', 'conditions': {}}
    for c in ('reward_dominant', 'balanced', 'choice_dominant'):
        val = RestlessTranscriptDataset(DATA, 'val', c)
        test = RestlessTranscriptDataset(DATA, 'test', c)
        va = val.tokens[:, 1::2].numpy()-1; a = test.tokens[:, 1::2].numpy()-1
        assert not set(val.audit.base_participant_id) & set(test.audit.base_participant_id)
        n, tmax = a.shape; rows = np.arange(n)
        freq = np.bincount(va[:, 1:].ravel(), minlength=4).astype(float)
        freq /= freq.sum()
        rate = float((va[:, 1:] == va[:, :-1]).mean())
        probs = {'uniform': np.full((n,tmax-1,4), .25),
                 'static_arm_frequency': np.broadcast_to(freq, (n,tmax-1,4)).copy(),
                 'static_stay_probability': np.full((n,tmax-1,4), (1-rate)/3),
                 'online_arm_frequency': np.empty((n,tmax-1,4)),
                 'online_stay_probability': np.empty((n,tmax-1,4))}
        transition = np.ones((4,4))
        np.add.at(transition, (va[:,:-1].ravel(),va[:,1:].ravel()), 1)
        transition /= transition.sum(-1,keepdims=True)
        probs['static_transition'] = transition[a[:,:-1]]
        probs['online_transition'] = np.empty((n,tmax-1,4))
        lag2_rate = float((va[:,2:]==va[:,:-2]).mean())
        probs['lag2_probability'] = np.full((n,tmax-1,4),(1-lag2_rate)/3)
        np.put_along_axis(probs['lag2_probability'][:,1:],a[:,:-2,None],lag2_rate,-1)
        probs['lag2_probability'][:,0] = probs['uniform'][:,0]
        np.put_along_axis(probs['static_stay_probability'], a[:, :-1, None], rate, -1)
        counts = np.ones((n,4)); repeats = np.ones(n); switches = np.ones(n)
        transition_counts = np.ones((n,4,4))
        counts[rows,a[:,0]] += 1
        for t in range(1,tmax):
            current = transition_counts[rows,a[:,t-1]]
            probs['online_transition'][:,t-1] = current/current.sum(-1,keepdims=True)
            probs['online_arm_frequency'][:,t-1] = counts/counts.sum(-1,keepdims=True)
            s = repeats/(repeats+switches)
            probs['online_stay_probability'][:,t-1] = (1-s[:,None])/3
            probs['online_stay_probability'][rows,t-1,a[:,t-1]] = s
            # Only update after this trial's predictive distribution is emitted.
            counts[rows,a[:,t]] += 1
            transition_counts[rows,a[:,t-1],a[:,t]] += 1
            same = a[:,t] == a[:,t-1]
            repeats += same; switches += ~same
        metrics = {k: score(p,a[:,1:]) for k,p in probs.items()}
        seeds = []
        for seed in (11,22,33):
            with np.load(f'outputs/gru_suite_20260907/{c}_choice_only_seed{seed}/predictions.npz') as z:
                assert np.array_equal(a,z['actions']) and np.array_equal(test.audit.base_participant_id,z['participants'])
                p = z['intact_choice_probabilities'].astype(float)[:,1:]
            p /= p.sum(-1,keepdims=True)
            seeds.append(score(p,a[:,1:]))
        metrics['gru_choice_only'] = {k: np.mean([s[k] for s in seeds],axis=0) for k in seeds[0]}
        strata = {}
        for name, mask in [('all',np.ones(n,bool)),('positive',test.audit.kernel_sign==1),('negative',test.audit.kernel_sign==-1)]:
            strata[name] = {model: {k: interval(v[mask]) for k,v in ms.items()} for model,ms in metrics.items()}
            strata[name]['paired_gru_minus_baseline_nll'] = {model: interval((metrics['gru_choice_only']['nll']-ms['nll'])[mask]) for model,ms in metrics.items() if model != 'gru_choice_only'}
        result['conditions'][c] = {'validation_arm_proportions':freq.tolist(),'validation_stay_probability':rate,'strata':strata}
        print(c, json.dumps({model:{k:float(v.mean()) for k,v in ms.items()} for model,ms in metrics.items()}),flush=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__': main()
