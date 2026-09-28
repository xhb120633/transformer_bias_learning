"""Validation-only Llama export using the original uncached evaluation path."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np


def reward_batches(original, donors, batch_size):
    """Fixed-size, equal-length batches with intact row zero in every batch."""
    assert batch_size >= 2
    unique = list(dict.fromkeys(int(x) for x in donors if int(x) != original))
    chunks = [unique[i:i + batch_size - 1] for i in range(0, len(unique), batch_size - 1)] or [[]]
    return [[original] + chunk + [original] * (batch_size - 1 - len(chunk)) for chunk in chunks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', required=True)
    ap.add_argument('--mode', choices=['full', 'choice', 'local'], required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--test-main', action='store_true', help='Frozen test prediction and primary donor only')
    args = ap.parse_args()
    assert args.batch_size >= 2
    import unsloth
    import torch
    from unsloth import FastLanguageModel
    from trl import DataCollatorForCompletionOnlyLM
    from mechcal.analysis.evaluate_centaur_reward_placeholder import _reward_positions
    name = 'reward_w' + args.weight
    group = 'weight_choice_only_sft_20260916' if args.mode == 'choice' else 'weight_llama_sft_20260916'
    data = args.root / 'data' / group / (name + '_val.jsonl')
    adapter = args.root / 'outputs' / group / (name + '_seed100')
    maps_path = args.root / 'data/weight_llama_sft_20260916' / (name + '_val_maps.npz')
    out = args.root / 'outputs/weight_llama_eval_uncached_20260917' / ('smoke' if args.smoke else 'production') / name / args.mode
    if args.test_main:
        assert args.mode in ('full', 'choice'), 'Local ablation is outside frozen main-test scope'
        data = args.root / 'data/llama_test_20260925/task_a' / args.mode / (name + '.jsonl')
        maps_path = args.root / 'data/llama_test_20260925/task_a' / (name + '_maps.npz')
        out = args.root / 'outputs/llama_test_20260925/task_a' / ('smoke' if args.smoke else 'production') / name / args.mode
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(x) for x in data.read_text().splitlines()]
    z = np.load(maps_path)
    maps = z['donor_mappings']
    ids = np.array([str(x['participant']) for x in records])
    assert len(records) == 250 and maps.shape == (20, 250)
    assert np.array_equal(ids, z['participants'].astype(str))
    assert all(np.array_equal(np.sort(m), np.arange(250)) and np.all(m != np.arange(250)) for m in maps)
    indices = np.arange(250)[args.shard::args.shards]
    if args.smoke:
        indices = indices[:2]
    if all((out / f'participant_{i:03d}.npz').exists() for i in indices):
        print('ALREADY_COMPLETE', flush=True)
        return
    assert (adapter / 'adapter_model.safetensors').is_file()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(adapter), max_seq_length=32768, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    collator = DataCollatorForCompletionOnlyLM(response_template=tokenizer(' <<').input_ids[1:], instruction_template=tokenizer('>>').input_ids[1:], tokenizer=tokenizer)
    choices = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    assert all(len(x) == 1 for x in choices)
    choices = [x[0] for x in choices]
    device = next(model.parameters()).device
    prepared = []
    for rec in records:
        encoded = tokenizer(rec['text'], return_offsets_mapping=True, truncation=False)
        offsets = encoded.pop('offset_mapping')
        batch = collator([encoded])
        pos = batch['labels'][0].ne(-100).nonzero().flatten()
        assert len(pos) == 200
        actions = np.array(['ABCD'.index(x) for x in re.findall(r'<<([ABCD])>>', rec['text'])])
        assert len(actions) == 200
        assert np.array_equal(batch['input_ids'][0, pos].numpy(), np.array(choices)[actions])
        rp = _reward_positions(rec['text'], offsets) if args.mode != 'choice' else []
        if args.mode != 'choice':
            assert len(rp) == 200
            assert all(int(pos[t]) < rp[t] < int(pos[t + 1]) for t in range(199))
        else:
            assert 'receive' not in rec['text'] and 'points' not in rec['text']
        prepared.append((batch['input_ids'], pos, rp, actions))

    @torch.inference_mode()
    def forward(tokens, positions):
        # Match evaluate_centaur_donor_rewards.py: no KV reuse, explicit mask.
        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                       use_cache=False, return_dict=True)
        scores = output.logits[0, positions].float()
        lp = scores.log_softmax(-1)[:, choices].cpu().numpy()
        del output
        return lp

    @torch.inference_mode()
    def forward_local_batch(tokens):
        # All rows have equal length: no padding, KV reuse, or cross-row attention.
        # Only the final two logits are needed; read the one preceding choice.
        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                       use_cache=False, return_dict=True, num_logits_to_keep=2)
        lp = output.logits[:, -2].float().log_softmax(-1)[:, choices].cpu().numpy()
        del output
        return lp

    checks = []
    started = time.time()
    for count, i in enumerate(indices):
        target = out / f'participant_{i:03d}.npz'
        if target.exists():
            continue
        cpu, pos, rp, actions = prepared[i]
        tokens = cpu.to(device)
        intact = forward(tokens, (pos - 1).to(device))
        result = dict(participant=ids[i], index=i, actions=actions, intact_logp=intact, donor_indices=maps[:, i])
        repeats = range(2 if args.smoke else 20)
        if args.mode == 'full':
            for mode in (('donor',) if args.test_main else ('donor', 'suffix_donor')):
                values = []
                for r in repeats:
                    changed = tokens.clone()
                    donor = prepared[maps[r, i]][0]
                    donor_rp = prepared[maps[r, i]][2]
                    start = 0 if mode == 'donor' else 150
                    changed[0, rp[start:]] = donor[0, donor_rp[start:]].to(device)
                    lp = forward(changed, (pos - 1).to(device))
                    boundary = 1 if mode == 'donor' else 151
                    assert np.allclose(lp[:boundary], intact[:boundary], atol=2e-3, rtol=0), 'Causal prefix changed'
                    values.append(lp[150:])
                result[mode + '_logp'] = np.stack(values)
        if args.mode == 'local':
            trials = [150, 175, 199] if args.smoke else list(range(150, 200))
            values = np.empty((len(repeats), len(trials), 4), np.float32)
            local_intact = np.empty((len(trials), 4), np.float32)
            for ti, t in enumerate(trials):
                stop = rp[t - 1]
                # Original evaluator includes the target token but reads the
                # preceding position. The causal mask hides the target token.
                end = int(pos[t]) + 1
                prefix = tokens[:, :end]
                original = int(tokens[0, stop])
                donor_tokens = [int(prepared[maps[r, i]][0][0, prepared[maps[r, i]][2][t - 1]])
                                for r in repeats]
                memo = {}
                for bi, rewards in enumerate(reward_batches(original, donor_tokens, args.batch_size)):
                    changed = prefix.repeat(args.batch_size, 1)
                    changed[:, stop] = torch.tensor(rewards, device=device)
                    assert torch.all((changed != prefix).sum(1) <= 1)
                    lp = forward_local_batch(changed)
                    if bi == 0:
                        local_intact[ti] = lp[0]
                    else:
                        assert np.allclose(lp[0], local_intact[ti], atol=.003, rtol=0), 'Intact batch row changed'
                    for ri, reward in enumerate(rewards):
                        if reward not in memo:
                            memo[reward] = lp[ri]
                    if ti == 0 and bi == 0:
                        future_changed = changed.clone()
                        future_changed[:, -1] = choices[0]
                        probe = forward_local_batch(future_changed)
                        error = float(np.max(np.abs(probe-lp)))
                        checks.append(error)
                        assert error < .003, f'Batched future-token causality error: {error}'
                for r, reward in enumerate(donor_tokens):
                    values[r, ti] = memo[reward]
            result['local_logp'] = values
            result['local_intact_logp'] = local_intact
            result['target_indices'] = np.array(trials)
        if count == 0:
            # Test causality with identical tensor shapes, without comparing
            # different-length low-precision GEMMs. Change target and all future
            # tokens; the preceding prediction must remain unchanged.
            changed = tokens.clone()
            changed[:, int(pos[150]):] = choices[0]
            probe = forward(changed, [int(pos[150]) - 1])[0]
            error = float(np.max(np.abs(probe - intact[150])))
            checks.append(error)
            print('SAME_SHAPE_CAUSAL_AUDIT', error, flush=True)
            assert error < .003, f'Future tokens changed prediction: {error}'
        assert all(np.isfinite(v).all() for k, v in result.items() if k.endswith('logp'))
        for k, v in result.items():
            if k.endswith('logp'):
                assert np.all(np.exp(v).sum(-1) <= 1.0001)
        result['inference'] = 'uncached_batched_local' if args.mode == 'local' else 'uncached_original_path'
        result['batch_size'] = args.batch_size if args.mode == 'local' else 1
        # Atomic publication allows interruption without a partial .npz being resumed.
        temporary = target.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **result)
        temporary.replace(target)
        print('PARTICIPANT_COMPLETE', args.weight, args.mode, int(i), 'seconds', round(time.time() - started, 1), flush=True)
    metadata = dict(mode=args.mode, weight=args.weight, split='test' if args.test_main else 'val', smoke=args.smoke,
                    operations=['intact', 'donor'] if args.test_main and args.mode == 'full' else None,
                    indices=indices.tolist(), shard=args.shard, shards=args.shards,
                    inference='uncached_batched_local' if args.mode == 'local' else 'uncached_original_path',
                    batch_size=args.batch_size if args.mode == 'local' else 1,
                    causal_logp_max_error=max(checks, default=None),
                    source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(), maps_sha256=hashlib.sha256(maps_path.read_bytes()).hexdigest(),
                    adapter=str(adapter), probability='ABCD log probabilities under full vocabulary; normalize only in explicit conditional analysis',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / f'COMPLETE_shard{args.shard}.json').write_text(json.dumps(metadata, indent=2))
    print('EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
