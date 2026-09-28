"""Repair length-dependent numerics in the fixed-model reward-operation sweep.

All operations use the intact sequence's tensor length. Suffix filler tokens
occur strictly after every scored target; the autoregressive mask prevents
them from changing the conditioning history. The explicit attention mask is
all ones, preserving the original uncached Unsloth kernel path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', required=True, choices=['000','010','030','050','070','090','100'])
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    import unsloth
    import torch
    from trl import DataCollatorForCompletionOnlyLM
    from unsloth import FastLanguageModel
    from mechcal.analysis.evaluate_centaur_reward_placeholder import _reward_positions
    from mechcal.analysis.evaluate_centaur_history_only import history_only_text

    name = 'reward_w' + a.weight
    records_path = a.root / 'data/weight_llama_sft_20260916' / (name + '_val.jsonl')
    adapter = a.root / 'outputs/weight_llama_sft_20260916' / (name + '_seed100')
    old = a.root / 'outputs/weight_llama_eval_uncached_20260917/production' / name / 'full'
    stage = 'smoke' if a.limit is not None else 'production'
    out = a.root / 'outputs/weight_llama_operation_fixedshape_20260924' / stage / name
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(s) for s in records_path.read_text().splitlines()]
    assert len(records) == 250 and (adapter/'adapter_model.safetensors').is_file()
    indices = np.arange(250)[a.shard::a.shards]
    if a.limit is not None:
        indices = indices[:a.limit]
    if all((out/f'participant_{i:03d}.npz').exists() for i in indices):
        print('ALREADY_COMPLETE', a.weight, a.shard, flush=True)
        return
    model, tok = FastLanguageModel.from_pretrained(model_name=str(adapter),
        max_seq_length=32768, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    tok.pad_token_id = 0
    tok.padding_side = 'right'
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tok(' <<').input_ids[1:],
        instruction_template=tok('>>').input_ids[1:], tokenizer=tok)
    choice_ids = [tok.encode(s, add_special_tokens=False) for s in 'ABCD']
    neutral_ids = tok.encode('50', add_special_tokens=False)
    assert all(len(x)==1 for x in choice_ids) and len(neutral_ids)==1
    choice_ids = [x[0] for x in choice_ids]
    neutral_id = neutral_ids[0]
    placeholder_id = tok.convert_tokens_to_ids('<|reserved_special_token_0|>')
    assert placeholder_id is not None and placeholder_id != tok.unk_token_id
    device = next(model.parameters()).device
    started = time.time()

    @torch.inference_mode()
    def forward(tokens, positions, length):
        assert tokens.shape[0] == 1 and tokens.shape[1] <= length
        positions = torch.as_tensor(positions, device=device)
        assert positions.max() < tokens.shape[1]
        padded = torch.zeros((1, length), dtype=tokens.dtype, device=device)
        padded[:, :tokens.shape[1]] = tokens
        result = model(input_ids=padded, attention_mask=torch.ones_like(padded),
                       use_cache=False, return_dict=True)
        lp = result.logits[0, positions].float().log_softmax(-1)[:, choice_ids].cpu().numpy()
        del result
        return lp

    for i in indices:
        target = out/f'participant_{i:03d}.npz'
        if target.exists():
            continue
        rec = records[int(i)]
        encoded = tok(rec['text'], return_offsets_mapping=True, truncation=False)
        offsets = encoded.pop('offset_mapping')
        batch = collator([encoded])
        positions = batch['labels'][0].ne(-100).nonzero().flatten()
        rp = _reward_positions(rec['text'], offsets)
        actions = np.array(['ABCD'.index(s) for s in re.findall(r'<<([ABCD])>>', rec['text'])])
        assert len(positions) == len(rp) == len(actions) == 200
        assert np.array_equal(batch['input_ids'][0,positions].numpy(),np.array(choice_ids)[actions])
        assert all(int(positions[t]) < rp[t] < int(positions[t+1]) for t in range(199))
        tokens = batch['input_ids'].to(device)
        fixed_length = tokens.shape[1]
        with np.load(old/f'participant_{i:03d}.npz') as z:
            assert str(z['participant']) == str(rec['participant'])
            intact = z['intact_logp'].astype(float)
            assert np.array_equal(z['actions'], actions)
        if i == indices[0]:
            fresh = forward(tokens,(positions-1).to(device),fixed_length)
            assert np.max(np.abs(fresh-intact)) < .003, 'Intact inference differs from prior audited run'
        neutral = tokens.clone();neutral[0,rp] = neutral_id
        placeholder = tokens.clone();placeholder[0,rp] = placeholder_id
        keep = torch.ones(tokens.shape[1],dtype=torch.bool,device=device)
        keep[rp] = False
        deleted = tokens[:,keep]
        moved = positions.to(device) - torch.searchsorted(torch.as_tensor(rp,device=device),positions.to(device))
        assert torch.equal(deleted[0,moved],tokens[0,positions.to(device)])
        history_text = history_only_text(rec['text'])
        history_batch = collator([tok(history_text,truncation=False)])
        history_positions = history_batch['labels'][0].ne(-100).nonzero().flatten()
        assert len(history_positions) == 200
        assert np.array_equal(history_batch['input_ids'][0,history_positions].numpy(),
                              np.array(choice_ids)[actions])
        history_tokens = history_batch['input_ids'].to(device)
        predictions = {'neutral50_logp':forward(neutral,(positions-1).to(device),fixed_length),
                       'placeholder_logp':forward(placeholder,(positions-1).to(device),fixed_length),
                       'delete_logp':forward(deleted,moved-1,fixed_length),
                       'history_only_logp':forward(history_tokens,(history_positions-1).to(device),fixed_length)}
        assert all(v.shape==(200,4) and np.isfinite(v).all() for v in predictions.values())
        first_choice_errors={k:float(np.max(np.abs(v[0]-intact[0]))) for k,v in predictions.items()}
        assert first_choice_errors['neutral50_logp'] < .003
        assert first_choice_errors['placeholder_logp'] < .003
        assert first_choice_errors['delete_logp'] < .003, first_choice_errors
        future_error = None
        if i == indices[0]:
            probe = deleted.clone()
            probe[:, int(moved[150]):] = choice_ids[0]
            scored = forward(probe,[int(moved[150])-1],fixed_length)[0]
            future_error = float(np.max(np.abs(scored-predictions['delete_logp'][150])))
            assert future_error < .003, ('Deletion future-prefix check', future_error)
            print('FIXED_SHAPE_AUDIT', first_choice_errors,
                  'future_error', future_error, 'length', fixed_length, flush=True)
        temporary = target.with_suffix('.partial.npz')
        np.savez_compressed(temporary,participant=str(rec['participant']),index=int(i),actions=actions,
            intact_logp=intact,first_choice_errors=json.dumps(first_choice_errors),
            fixed_length=fixed_length,delete_length=deleted.shape[1],
            history_only_length=history_tokens.shape[1],
            future_error=np.nan if future_error is None else future_error,**predictions)
        temporary.replace(target)
        print('COMPLETE',a.weight,int(i),'elapsed_s',round(time.time()-started,1),flush=True)
    (out/f'COMPLETE_shard{a.shard}.json').write_text(json.dumps(dict(weight=a.weight,
        shard=a.shard,shards=a.shards,limit=a.limit,indices=indices.tolist(),split='val',
        operations=['neutral50','reserved_placeholder','delete_reward_tokens','history_only_rewrite'],
        adapter=str(adapter),inference='uncached_fixed_shape_future_suffix',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    main()
