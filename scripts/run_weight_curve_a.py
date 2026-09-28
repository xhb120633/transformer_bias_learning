"""Generator-only weight sweep. Explore validation; leave test unopened by analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mechcal.data.build_restless_dataset import _load_config, _make_generator_config
from mechcal.generators.four_armed_restless import (
    MechanismCondition, calibrate_q_pairwise_scale, generate_reward_schedule,
    generate_potential_rewards, generate_sessions,
)
from mechcal.analysis.generator_oracle_donor import oracle_trial_nll

WEIGHTS = (0., .1, .3, .5, .7, .9, 1.)


def probabilities(q, previous, data, scale):
    z = (q - q.mean(axis=1, keepdims=True)) / scale
    logits = data['beta_reward'][:, None] * z
    if previous is not None:
        logits[np.arange(len(q)), previous] += data['beta_kernel']
    winners = np.isclose(logits, logits.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
    eps = data['decision_noise'][:, None]
    return eps / 4 + (1 - eps) * winners / winners.sum(axis=1, keepdims=True)


def replay(data, rewards, scale, local=False):
    actions = data['action']
    n, tmax = actions.shape
    q = np.full((n, 4), 50., dtype=float)
    rows = np.arange(n)
    result = np.empty((n, tmax, 4))
    for t in range(tmax):
        altered = q.copy()
        if local and t:
            altered[rows, actions[:, t-1]] += data['alpha'] * (rewards[:, t-1] - data['reward'][:, t-1])
        result[:, t] = probabilities(altered, actions[:, t-1] if t else None, data, scale)
        update_rewards = data['reward'] if local else rewards
        q[rows, actions[:, t]] += data['alpha'] * (update_rewards[:, t] - q[rows, actions[:, t]])
    return result


def nll(p, a):
    return -np.log(np.take_along_axis(p, a[..., None], axis=2)[..., 0])


def derangements(n, count, seed):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            result.append(p)
    return np.stack(result)


def estimate(x, mask):
    x = np.asarray(x)[mask]
    rng = np.random.default_rng(20260915)
    boot = x[rng.integers(len(x), size=(2000, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())


def analyze(data, scale, mappings):
    p = replay(data, data['reward'], scale)
    assert np.max(np.abs(p - data['choice_probability'])) < 1e-6
    base = nll(p, data['action'])
    legacy = oracle_trial_nll(data['action'], data['reward'], data['alpha'], data['beta_reward'], data['beta_kernel'], scale, data['decision_noise'])
    assert np.max(np.abs(legacy - base)) < 1e-10
    arrays = dict(participants=data['base_participant_id'], kernel_sign=data['kernel_sign'], intact_nll=base,
                  donor_mappings=mappings, intact_probabilities=p)
    metrics = dict(intact_nll=base[:, 1:].mean(axis=1),
                   stay=(data['action'][:, 1:] == data['action'][:, :-1]).mean(axis=1),
                   lag2=(data['action'][:, 2:] == data['action'][:, :-2]).mean(axis=1),
                   mean_reward=data['reward'].mean(axis=1))
    for local, name in ((False, 'donor'), (True, 'local')):
        delta, tv, expected = [], [], []
        for mapping in mappings:
            perturbed = replay(data, data['reward'][mapping], scale, local)
            assert np.array_equal(perturbed[:, 0], p[:, 0])
            if np.all(data['beta_reward'] == 0):
                assert np.array_equal(perturbed, p)
            if local:
                # Independently verify a one-reward intervention by full recursive replay.
                altered = data['reward'].copy()
                altered[:, 98] = data['reward'][mapping, 98]
                check = replay(data, altered, scale)
                assert np.max(np.abs(check[:, 99] - perturbed[:, 99])) < 1e-10
            delta.append(nll(perturbed, data['action']) - base)
            tv.append(.5 * np.abs(perturbed - p).sum(axis=2))
            expected.append((p * np.log(p / perturbed)).sum(axis=2))
        for metric, values in (('delta_nll', delta), ('tv', tv), ('expected_delta_nll', expected)):
            values = np.stack(values)
            arrays[f'{name}_{metric}'] = values
            metrics[f'{name}_{metric}'] = values[:, :, 1:].mean(axis=(0, 2))
    result = {}
    for group, mask in dict(all=np.ones(len(p), bool), positive=data['kernel_sign'] > 0, negative=data['kernel_sign'] < 0).items():
        result[group] = {'n': int(mask.sum()), **{k: estimate(v, mask) for k, v in metrics.items()}}
    return result, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = _load_config(Path('configs/data/restless_pooled_eps10_1000.yaml'))
    cfg = _make_generator_config(raw)
    counts = dict(train=12, val=12, test=12) if args.smoke else dict(train=1000, val=250, test=250)
    # Fresh ranges, disjoint from the historical benchmark and from each other.
    starts = dict(train=11000000, val=12000000, test=13000000)
    seed = 20260915
    scale = calibrate_q_pairwise_scale(cfg, list(range(9900000, 9900256)), 256, seed + 700000)
    protocol = dict(weights=list(WEIGHTS), generator=raw['generator'], counts=counts,
                    schedule_starts=starts, calibration_start=9900000, q_scale=scale, seed=seed,
                    analysis_split='val', scored_trials='2-200', donor_repeats=20,
                    pairing='same participant parameters, schedules, potential rewards and choice RNG across weights',
                    local='only reward t-1 independently replaced at each target t; no accumulation',
                    expected_delta_nll='KL(intact oracle || perturbed oracle), conditional on intact history',
                    limitations='conditional fixed-choice response, not an environment intervention; CIs conditional on 20 donor maps')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    summary = {}
    hashes = {}
    for si, (split, count) in enumerate(counts.items()):
        schedules = {s: generate_reward_schedule(cfg, s) for s in range(starts[split], starts[split] + count)}
        hashes[split] = {hashlib.sha256(v.tobytes()).hexdigest() for v in schedules.values()}
        assert len(hashes[split]) == count
        for other in hashes:
            if other != split:
                assert not hashes[split] & hashes[other]
        potential = {s: generate_potential_rewards(cfg, v, s) for s, v in schedules.items()}
        reference = None
        mappings = derangements(count, 20, seed + si)
        for w in WEIGHTS:
            name = f'reward_w{round(w*100):03d}'
            dest = args.output / name
            dest.mkdir(exist_ok=True)
            data = generate_sessions(cfg, MechanismCondition(name, 1-w, 1-w), schedules, count, seed + si*10000, scale, potential)
            data['base_participant_id'] = data['session_id'].copy()
            data['session_id'] = np.asarray([f'{s}_{name}' for s in data['session_id']])
            if reference is not None:
                for key in ('base_participant_id', 'alpha', 'kernel_sign', 'policy_scale', 'schedule_seed', 'potential_reward', 'latent_reward_mean'):
                    assert np.array_equal(data[key], reference[key]), key
            else:
                reference = data
            np.savez_compressed(dest / f'{split}_000.npz', **data)
            if split == 'val':
                result, arrays = analyze(data, scale, mappings)
                summary[name] = dict(weight=w, groups=result)
                np.savez_compressed(dest / 'validation_assay.npz', **arrays)
                (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f'{split} {name}: generated, pairing audited' + ('; oracle audited' if split == 'val' else ''), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, ('stay', 'lag2', 'donor_delta_nll', 'local_delta_nll', 'donor_tv', 'local_tv')):
        for group, label in [('all', 'All'), ('positive', 'Positive kernel'), ('negative', 'Negative kernel')]:
            ys = [v['groups'][group][metric]['mean'] for v in summary.values()]
            ci = np.array([v['groups'][group][metric]['ci95'] for v in summary.values()])
            ax.plot(WEIGHTS, ys, 'o-', label=label)
            ax.fill_between(WEIGHTS, ci[:, 0], ci[:, 1], alpha=.12)
        ax.set(xlabel='Reward policy weight', ylabel=metric.replace('_', ' '))
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Generator reference curves | validation only | trials 2-200 (lag2: 3-200)')
    fig.savefig(args.output / 'reference_curves.png', dpi=180)
    fig.savefig(args.output / 'reference_curves.svg')
    (args.output / 'COMPLETE.json').write_text(json.dumps(dict(complete=True, audits_passed=True, test_analyzed=False, smoke=args.smoke)), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
