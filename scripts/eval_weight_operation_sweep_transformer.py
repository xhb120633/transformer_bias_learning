"""Seven-weight, validation-only reward-operation sweep for full-input Transformers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechcal.models import CausalTransformer, TransformerConfig
from run_reference_followup import ROOT, load, WEIGHTS


@torch.inference_mode()
def probabilities(model, tokens, operation, batch_size=32):
    rows = []
    for batch in tokens.split(batch_size):
        x = batch[:, :-1].cuda()
        mask = None
        if operation == 'zero_embedding':
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, 2::2] = True
        logits = model(x, zero_token_embedding_mask=mask).float()
        positions = slice(0, 400, 2) if operation != 'delete' else slice(0, 200)
        rows.append(logits[:, positions, 1:5].softmax(-1).cpu().numpy())
    return np.concatenate(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='one checkpoint directory name, for smoke/resume')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path('outputs/weight_neural_full_20260916')
    out = Path('outputs/weight_operation_sweep_transformer_20260923')
    out.mkdir(parents=True, exist_ok=True)
    runs = [root / args.run] if args.run else [root / f'transformer_reward_w{round(w*100):03d}_full_seed{s}'
                                             for w in WEIGHTS for s in (11, 22, 33)]
    for run in runs:
        target = out / f'{run.name}.npz'
        if target.exists():
            continue
        _, _, code, mode, seed = run.name.split('_')
        assert mode == 'full'
        w = int(code[1:]) / 100
        data = load(w)
        actions = data['action'][:, 150:]
        tokens = torch.tensor(data['tokens'].astype(np.int64))
        assert tokens.shape == (250, 401)
        ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        model = CausalTransformer(TransformerConfig(**ck['model_config'])).cuda().eval()
        model.load_state_dict(ck['model_state'])
        original = probabilities(model, tokens, 'intact')
        with np.load(run / 'assay_full.npz') as saved:
            np.testing.assert_array_equal(saved['participants'], data['base_participant_id'])
            np.testing.assert_allclose(original, saved['intact_probabilities'], atol=1e-6)
            donor = saved['donor_probabilities'].astype(float)
        changed = tokens.clone()
        changed[:, 2::2] = 55
        neutral = probabilities(model, changed, 'neutral50')
        zero = probabilities(model, tokens, 'zero_embedding')
        deletion = probabilities(model, torch.cat((tokens[:, :1], tokens[:, 1::2]), dim=1), 'delete')
        assert original.shape == neutral.shape == zero.shape == deletion.shape == (250, 200, 4)
        assert np.allclose(original[:, 0], neutral[:, 0], atol=1e-6)
        assert np.allclose(original[:, 0], zero[:, 0], atol=1e-6)
        base = -np.log(np.take_along_axis(original[:, 150:], actions[..., None], -1)[..., 0]).mean(1)
        metrics = {}
        for name, p in [('donor', donor), ('neutral50', neutral[:, 150:]),
                        ('zero_embedding', zero[:, 150:]), ('delete', deletion[:, 150:])]:
            if name == 'donor':
                losses = -np.log(np.take_along_axis(p, actions[None, ..., None], -1)[..., 0]).mean((0, 2))
            else:
                losses = -np.log(np.take_along_axis(p, actions[..., None], -1)[..., 0]).mean(1)
            metrics[name + '_delta_nll'] = losses - base
        np.savez_compressed(target, participants=data['base_participant_id'],
                            intact_nll=base, **metrics)
        print('COMPLETE', run.name, {k: round(float(v.mean()), 5) for k, v in metrics.items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    if not args.run:
        (out / 'COMPLETE.json').write_text(json.dumps({'weights': list(WEIGHTS), 'seeds': [11,22,33],
            'split': 'val', 'score_trials': '151-200', 'test_used': False, 'runs': len(runs)}, indent=2))


if __name__ == '__main__':
    main()
